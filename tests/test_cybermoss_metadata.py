"""MOSS capture and resource ancestry regression tests; no model/GPU needed."""
import ast
import asyncio
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    '_moss_capture_fixture', ROOT / 'tests/test_negpip_prompt_routing.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def graph(loader='UNETLoader', refine=False):
    prompt = {
        'unused': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'wrong.safetensors'}},
        'save': {'class_type': 'CyberdeliaSaveImageWithMetaData', 'inputs': {'images': ['decode', 0]}},
        'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}},
        'sample': {'class_type': 'CyberMOSSSampler', 'inputs': {
            'model': ['lora', 0], 'positive': ['positive', 0], 'negative': ['negative', 0],
            'latent_image': ['latent', 0], 'seed': 42, 'steps': 30, 'cfg': 3.,
            'sampler': 'euler_2m', 'schedule_mode': 'reference', 'preset': 'full_reference',
            'denoise': 1., 'skip_layers': '22', 'restart_frac': 0., 'sigma_r': .65,
            'plunge': False, 'detail': 0., 'eta0': 0., 'sigma_gate': .1, 'contraction': 1.}},
        'model': {'class_type': loader, 'inputs': {'unet_name': 'moss/CyberMOSS_merge.safetensors'}},
        'lora': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['model', 0], 'lora_name': 'portrait.safetensors', 'strength_model': .75}},
        'positive': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'editorial portrait'}},
        'negative': {'class_type': 'CLIPTextEncode', 'inputs': {'text': 'watermark'}},
        'latent': {'class_type': 'CyberMOSSEmptyLatent', 'inputs': {
            'size': 'M (~1.5 MP)', 'resolution': '1024x1536 (2:3)', 'batch_size': 1}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'flux2-vae.safetensors'}},
    }
    if refine:
        prompt['decode']['inputs']['samples'] = ['refine', 0]
        prompt['refine'] = {'class_type': 'CyberMOSSSampler', 'inputs': {
            **prompt['sample']['inputs'], 'model': ['refine_model', 0],
            'latent_image': ['sample', 0], 'steps': 12, 'seed': 999, 'cfg': 1.,
            'schedule_mode': 'simple', 'denoise': .35, 'preset': 'turbo_reference'}}
        prompt['refine_model'] = {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'refiner.safetensors'}}
    return prompt


class CyberMOSSMetadataTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = sys.modules.copy()
        self.capture = fixture._load_capture()
        package = self.capture.__package__
        self.meta = self.capture.MetaField
        self.samplers = fixture._load_module(package + '.defs.samplers', ROOT / 'modules/defs/samplers.py')
        fixture._load_module(package + '.defs.validators', ROOT / 'modules/defs/validators.py')
        formatters = sys.modules[package + '.defs.formatters']
        tree = ast.parse((ROOT / 'modules/defs/captures.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'formatters':
                for alias in node.names:
                    setattr(formatters, alias.name, lambda value, *a, **kw: 'hash-' + value)
        base = fixture._load_module(package + '.defs.captures', ROOT / 'modules/defs/captures.py')
        self.capture.CAPTURE_FIELD_LIST = base.CAPTURE_FIELD_LIST
        self.extension = fixture._load_module(package + '.defs.ext.cybermoss_sampler', ROOT / 'modules/defs/ext/cybermoss_sampler.py')
        multi = fixture._load_module(package + '.defs.ext.comfyui_multigpu', ROOT / 'modules/defs/ext/comfyui_multigpu.py')
        self.capture.CAPTURE_FIELD_LIST.update(self.extension.CAPTURE_FIELD_LIST)
        self.capture.CAPTURE_FIELD_LIST.update(multi.CAPTURE_FIELD_LIST)
        self.samplers.SAMPLERS.update(self.extension.SAMPLERS)
        self.Trace = fixture._load_module(package + '.trace', ROOT / 'modules/trace.py').Trace
        self.capture.Trace = self.Trace

    def tearDown(self):
        for name in list(sys.modules):
            if name not in self.saved_modules:
                del sys.modules[name]
        sys.modules.update(self.saved_modules)

    def capture_graph(self, prompt):
        hook = sys.modules[self.capture.__package__ + '.hook']
        hook.current_prompt = prompt
        hook.current_extra_data = {}
        hook.current_save_image_node_id = 'save'
        hook.prompt_executer = None
        for node in prompt.values():
            self.capture.NODE_CLASS_MAPPINGS[node['class_type']] = type('HostNode', (), {})
        self.capture.NODE_CLASS_MAPPINGS['CyberMOSSEmptyLatent'] = type('Latent', (), {
            'INPUT_TYPES': staticmethod(lambda: {'required': {'size': ([], {'cybermoss_resolutions': {
                'M (~1.5 MP)': ['1024x1536 (2:3)'], 'L (~2.0 MP)': ['1184x1776 (2:3)']}})}})})
        execution = types.ModuleType('execution')
        async def get_input_data(inputs, *args):
            # Match the host without execution_list: linked values are missing.
            return ({k: [None] if isinstance(v, list) else [v] for k, v in inputs.items()}, {}, {})
        execution.get_input_data = get_input_data
        sys.modules['execution'] = execution
        sys.modules['comfy_execution'] = types.ModuleType('comfy_execution')
        graph_module = types.ModuleType('comfy_execution.graph')
        graph_module.DynamicPrompt = lambda prompt: prompt
        sys.modules[graph_module.__name__] = graph_module
        captured = asyncio.run(self.capture.Capture.get_inputs())
        trace = self.Trace.trace('save', prompt)
        sampler = self.Trace.find_sampler_node_id(trace)
        primary = self.Trace.filter_inputs_by_trace_tree(captured, self.Trace.trace(sampler, prompt), True)
        all_inputs = self.Trace.filter_inputs_by_trace_tree(captured, trace, True)
        info = self.capture.Capture.gen_pnginfo_dict(primary, all_inputs, prompt,
            sampler_node_id=sampler, active_trace_tree=trace)
        return info, sampler

    def test_model_hash_lora_vae_and_settings_survive_moss_ancestry(self):
        for loader in ('UNETLoader', 'UNETLoaderMultiGPU', 'UNETLoaderDisTorch2MultiGPU'):
            with self.subTest(loader=loader):
                info, sampler = self.capture_graph(graph(loader))
                self.assertEqual(sampler, 'sample')
                self.assertEqual(info['Model'], 'CyberMOSS_merge')
                self.assertEqual(info['Model hash'], 'hash-moss/CyberMOSS_merge.safetensors')
                self.assertEqual(info['VAE'], 'flux2-vae')
                self.assertEqual(info['Steps'], '30')
                self.assertEqual(info['Seed'], '42')
                self.assertEqual(info['CFG scale'], '3')
                self.assertEqual(info['Sampler'], 'CyberMOSS Euler 2M')
                self.assertEqual(info['Schedule type'], 'MOSS Reference')
                self.assertEqual(info['Size'], '1024x1536')
                self.assertEqual(info['Positive prompt'], 'editorial portrait <lora:portrait:0.75>')
                self.assertEqual(info['Negative prompt'], 'watermark')
                self.assertEqual(info['MOSS preset'], 'full_reference')
                self.assertEqual(info['MOSS skip layers'], '22')
                self.assertEqual(info['MOSS shift'], '5.0')
                parameters = self.capture.Capture.gen_parameters_str(info)
                self.assertIn('portrait', parameters)
                self.assertNotIn('wrong', parameters)

    def test_refinement_keeps_primary_generation_model_and_settings(self):
        info, sampler = self.capture_graph(graph(refine=True))
        self.assertEqual(sampler, 'sample')
        self.assertEqual(info['Model'], 'CyberMOSS_merge')
        self.assertEqual(info['Steps'], '30')
        self.assertEqual(info['Seed'], '42')
        self.assertEqual(info['MOSS preset'], 'full_reference')

    def test_linked_seed_steps_cfg_and_details(self):
        prompt = graph()
        prompt['seed'] = {'class_type': 'Seed (rgthree)', 'inputs': {'seed': 123456789}}
        prompt['steps'] = {'class_type': 'PrimitiveInt', 'inputs': {'value': 24}}
        prompt['cfg'] = {'class_type': 'PrimitiveFloat', 'inputs': {'value': 4.5}}
        prompt['detail'] = {'class_type': 'PrimitiveFloat', 'inputs': {'value': .4}}
        prompt['context'] = {'class_type': 'Context Big (rgthree)', 'inputs': {'seed': ['seed', 0]}}
        prompt['sample']['inputs'].update(seed=['context', 8], steps=['steps', 0], cfg=['cfg', 0], detail=['detail', 0])
        info, _ = self.capture_graph(prompt)
        self.assertEqual(info['Seed'], '123456789')
        self.assertEqual(info['Steps'], '24')
        self.assertEqual(info['CFG scale'], '4.5')
        self.assertEqual(info['MOSS detail'], '0.4')

    def test_runtime_batch_values_override_graph_and_are_not_links(self):
        obj = graph()['sample']
        select = self.extension.CAPTURE_FIELD_LIST['CyberMOSSSampler'][self.meta.SEED]['selector']
        prompt = {'1': {'inputs': {'value': 999}}}
        for index, expected in enumerate((1, 2)):
            self.assertEqual(select('sample', obj, prompt, {}, None, ({'seed': [1, 2]},), batch_index=index), expected)
        details = self.extension.get_sampler_details('sample', obj, {}, {}, None,
            ({'preset': ['turbo', 'full'], 'detail': [.1, .7]},), batch_index=1)
        self.assertEqual(details['MOSS preset'], 'full')
        self.assertEqual(details['MOSS detail'], .7)

    def test_turbo_schedules_and_denoise_are_not_reported_as_reference(self):
        for mode, expected in (('cybermoss', 'CyberMOSS'), ('simple', 'Simple')):
            prompt = graph()
            prompt['sample']['inputs'].update(cfg=1., preset='turbo', schedule_mode=mode,
                denoise=.45, steps=8, sampler='euler', skip_layers='')
            info, _ = self.capture_graph(prompt)
            self.assertEqual(info['Schedule type'], expected)
            self.assertEqual(info['CFG scale'], '1')
            self.assertEqual(info['Denoising strength'], .45)

    def test_stale_resolution_tier_uses_active_bucket(self):
        prompt = graph()
        prompt['latent']['inputs']['size'] = 'L (~2.0 MP)'
        info, _ = self.capture_graph(prompt)
        self.assertEqual(info['Size'], '1184x1776')

    def test_missing_negative_recovers_original_negpip_prompts(self):
        prompt = graph()
        prompt['sample']['inputs'].pop('negative')
        prompt['sample']['inputs'].update(model=['negpip', 0], positive=['negpip', 1], cfg=1.)
        prompt['negpip'] = {'class_type': 'ZImageNegPipPrompt', 'inputs': {
            'model': ['lora', 0], 'positive': 'MOSS portrait', 'negative': 'bad anatomy'}}
        # NegPiP 2.2 uses the same node id for its automatic MOSS model detection.
        positive, negative = self.capture._find_prompt_texts(prompt, None, sampler_node_id='sample')
        self.assertEqual((positive, negative), ('MOSS portrait', 'bad anatomy'))


if __name__ == '__main__':
    unittest.main()
