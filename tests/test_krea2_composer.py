"""Composer routing and cached passive metadata; no GPU/model needed."""
import asyncio
import importlib.util
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_composer_capture_fixture', ROOT / 'tests/test_negpip_prompt_routing.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def graph(negative=True, cls='KSampler'):
    inputs = {'positive': ['composer', 1], 'model': ['composer', 0], 'steps': 8, 'cfg': 1}
    if negative:
        inputs['negative'] = ['composer', 4]
    return {'sampler': {'class_type': cls, 'inputs': inputs},
            'composer': {'class_type': 'Krea2TwoCharacterComposer', 'inputs': {
                'base_prompt': 'raw scene {outfit}', 'negative_prompt': 'raw negative',
                'character_1_trigger': 'trigger_a', 'character_1_prompt': 'a woman',
                'character_2_trigger': 'trigger_b', 'character_2_prompt': 'a man',
                'character_1_lora': 'a.safetensors', 'character_1_strength': .75,
                'character_2_lora': 'b.safetensors', 'character_2_strength': 1.25}}}


def payload(suffix=''):
    return {'version': 1, 'scene_prompt': 'Two people in a garden'+suffix,
            'encoded_scene_prompt': 'Two people in a garden, (blurry:-1)',
            'character_prompts': ['trigger_a, woman wearing a navy uniform'+suffix,
                                  'trigger_b, man wearing a navy uniform'+suffix],
            'negative_prompt': 'blurry, watermark'+suffix,
            'loras': [[{'name': 'a.safetensors', 'strength': .75}], [{'name': 'b.safetensors', 'strength': 1.25}]],
            'layout': 'Free boxes (0,0,40,100; 50,0,50,100)', 'separation': 'Balanced',
            'lora_edge_softness': 0, 'base_strength': 1, 'negpip': True,
            'width': 1152, 'height': 1440}


class ComposerMetadataTests(unittest.TestCase):
    def setUp(self):
        self.capture = fixture._load_capture()
        self.helper = sys.modules[self.capture.__package__+'.composer_metadata']
        self.capture._resolved_node_texts.clear()
        self.capture._resolved_composer_metadata.clear()

    def test_original_prompts_fallback_without_new_composer(self):
        positive, negative = self.capture._find_prompt_texts(graph(), None, sampler_node_id='sampler')
        self.assertIn('Scene: raw scene {outfit}', positive)
        self.assertIn('Character 1: trigger_a, a woman', positive)
        self.assertIn('Character 2: trigger_b, a man', positive)
        self.assertEqual(negative, 'raw negative')

    def test_diagnostic_output_does_not_override_conditioning_roles(self):
        self.capture._resolved_node_texts.update({'composer:5': 'Scene: compiled diagnostics', 'composer': 'Scene: compiled diagnostics'})
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        positive, negative = self.capture._find_prompt_texts(graph(), None, sampler_node_id='sampler')
        self.assertIn('navy uniform', positive)
        self.assertNotIn('{outfit}', positive)
        self.assertNotIn('compiled diagnostics', positive)
        self.assertNotIn('(blurry:-1)', positive)
        self.assertEqual(negative, 'blurry, watermark')
        for slot in (0, 2, 3, 5):
            self.assertIsNone(self.capture._follow_conditioning_to_clip_text(['composer', slot], graph(), None))

    def test_cyberkrea_with_disconnected_negative_recovers_negpip_source(self):
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        positive, negative = self.capture._find_prompt_texts(graph(False, 'CyberKreaSampler'), None, sampler_node_id='sampler')
        self.assertIn('Character 2:', positive)
        self.assertEqual(negative, 'blurry, watermark')

    def test_context_routes_do_not_mix_positive_and_negative(self):
        prompt = graph()
        prompt['context'] = {'class_type': 'Context Big (rgthree)', 'inputs': {'positive': ['composer', 1], 'negative': ['composer', 4]}}
        prompt['sampler']['inputs'].update(positive=['context', 4], negative=['context', 5])
        self.capture._resolved_node_texts['context:15'] = 'irrelevant cached text'
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        positive, negative = self.capture._find_prompt_texts(prompt, None, sampler_node_id='sampler')
        self.assertEqual(positive, self.helper.positive_text(payload()))
        self.assertEqual(negative, 'blurry, watermark')

    def test_advanced_sampler_and_guider(self):
        prompt = graph()
        prompt['guider'] = {'class_type': 'CFGGuider', 'inputs': {'positive': ['composer', 1], 'negative': ['composer', 4]}}
        prompt['sampler'] = {'class_type': 'SamplerCustomAdvanced', 'inputs': {'guider': ['guider', 0]}}
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        positive, negative = self.capture._find_prompt_texts(prompt, None, sampler_node_id='sampler')
        self.assertEqual(positive, self.helper.positive_text(payload()))
        self.assertEqual(negative, 'blurry, watermark')

    def test_basic_guider_recovers_negpip_negative_from_positive_composer(self):
        prompt = graph(False)
        prompt['guider'] = {'class_type': 'BasicGuider', 'inputs': {'conditioning': ['composer', 1]}}
        prompt['sampler'] = {'class_type': 'SamplerCustomAdvanced', 'inputs': {'guider': ['guider', 0]}}
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        positive, negative = self.capture._find_prompt_texts(prompt, None, sampler_node_id='sampler')
        self.assertEqual(positive, self.helper.positive_text(payload()))
        self.assertEqual(negative, 'blurry, watermark')

    def test_batch_index_selects_matching_runtime_prompts(self):
        self.capture._resolved_composer_metadata['composer'] = [payload(' first'), payload(' second')]
        pair = self.capture._find_prompt_texts(graph(), None, batch_index=1, sampler_node_id='sampler')
        self.assertIn('uniform second', pair[0])
        self.assertEqual(pair[1], 'blurry, watermark second')
        self.capture._clear_resolved_texts()
        self.assertEqual(self.capture._resolved_composer_metadata, {})

    def test_cached_payload_reader_handles_real_output_wrapping_and_ignores_report(self):
        cached = [[object()], [[['tensor stand-in', {self.helper.METADATA_KEY: payload()}]]], [object()], [], [], ['human report']]
        self.assertEqual(self.helper.cached_payloads(cached), [payload()])
        cached[1].append([['another tensor', {self.helper.METADATA_KEY: payload(' second')}]])
        self.assertEqual(len(self.helper.cached_payloads(cached)), 2)
        self.assertEqual(self.helper.cached_payloads([[], [], [], [], [], ['human report']]), [])

    def test_selectors_preserve_both_lora_resources_and_regions(self):
        name = self.capture.__package__+'.defs.ext.krea2_composer'
        spec = importlib.util.spec_from_file_location(name, ROOT/'modules/defs/ext/krea2_composer.py')
        extension = importlib.util.module_from_spec(spec)
        sys.modules[name] = extension
        spec.loader.exec_module(extension)
        self.capture._resolved_composer_metadata['composer'] = [payload(' first'), payload(' second')]
        fields = extension.CAPTURE_FIELD_LIST['Krea2TwoCharacterComposer']
        meta = self.capture.MetaField
        select = lambda key: fields[key]['selector']('composer', graph()['composer'], graph(), {}, None, ({},), batch_index=1)
        self.assertIn('uniform second', select(meta.POSITIVE_PROMPT))
        self.assertEqual(select(meta.NEGATIVE_PROMPT), 'blurry, watermark second')
        self.assertEqual(select(meta.LORA_MODEL_NAME), ['a.safetensors', 'b.safetensors'])
        self.assertEqual(select(meta.LORA_STRENGTH_MODEL), [.75, 1.25])
        self.assertEqual(select(meta.IMAGE_WIDTH), 1152)
        self.assertIn('a.safetensors @ 0.75', select(meta.CUSTOM_PARAMETERS)['Krea character 1 LoRAs'])
        self.assertTrue(all(field['per_image'] for field in fields.values()))

    def test_async_cache_rehydrates_metadata_without_reexecuting_composer(self):
        prompt = graph()
        prompt['sampler']['inputs'].update(seed=123, sampler_name='euler', scheduler='beta')
        prompt['save'] = {'class_type': 'CyberdeliaSaveImageWithMetaData', 'inputs': {'images': ['sampler', 0]}}
        hook = sys.modules[self.capture.__package__+'.hook']
        hook.current_prompt = prompt
        hook.current_extra_data = {}
        hook.current_save_image_node_id = 'save'
        cached = [[], [[['tensor', {self.helper.METADATA_KEY: payload()}]]], [], [], [], ['diagnostic report']]
        class Cache:
            async def get(self, node_id):
                return types.SimpleNamespace(outputs=cached) if node_id == 'composer' else None
        hook.prompt_executer = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=Cache()))
        package = self.capture.__package__
        fixture._load_module(package+'.defs.samplers', ROOT/'modules/defs/samplers.py')
        trace = fixture._load_module(package+'.trace', ROOT/'modules/trace.py')
        self.capture.Trace = trace.Trace
        extension = fixture._load_module(package+'.defs.ext.krea2_composer', ROOT/'modules/defs/ext/krea2_composer.py')
        extension.calc_lora_hash = lambda name: 'hash-'+name
        self.capture.CAPTURE_FIELD_LIST.update(extension.CAPTURE_FIELD_LIST)
        self.capture.NODE_CLASS_MAPPINGS['Krea2TwoCharacterComposer'] = type('Composer', (), {})
        self.capture.NODE_CLASS_MAPPINGS['KSampler'] = type('Sampler', (), {})
        self.capture.CAPTURE_FIELD_LIST['KSampler'] = {self.capture.MetaField.STEPS: {'field_name': 'steps'}, self.capture.MetaField.SEED: {'field_name': 'seed'}}
        execution = types.ModuleType('execution')
        async def get_input_data(inputs, *args):
            return ({key: [value] for key, value in inputs.items()}, {}, {})
        execution.get_input_data = get_input_data
        sys.modules['execution'] = execution
        graph_module = types.ModuleType('comfy_execution.graph')
        graph_module.DynamicPrompt = lambda prompt: prompt
        sys.modules['comfy_execution'] = types.ModuleType('comfy_execution')
        sys.modules[graph_module.__name__] = graph_module
        for _ in range(2):
            captured = asyncio.run(self.capture.Capture.get_inputs())
            meta = self.capture.MetaField
            self.assertIn('navy uniform', captured[meta.POSITIVE_PROMPT][0][1])
            self.assertEqual(captured[meta.NEGATIVE_PROMPT][0][1], 'blurry, watermark')
            self.assertEqual([row[1] for row in captured[meta.LORA_MODEL_NAME]], ['a.safetensors', 'b.safetensors'])
            self.assertEqual(captured[meta.IMAGE_WIDTH][0][1], 1152)
            result = self.capture.Capture.gen_pnginfo_dict(captured, defaultdict(list), prompt, sampler_node_id='sampler')
            self.assertIn('Character 2: trigger_b, man', result['Positive prompt'])
            self.assertEqual(result['Negative prompt'], 'blurry, watermark')
            self.assertIn('navy uniform', result['Krea character 1 prompt'])
            self.assertIn('a.safetensors @ 0.75', result['Krea character 1 LoRAs'])

    def test_generated_pnginfo_keeps_three_prompts_and_original_negative(self):
        self.capture._resolved_composer_metadata['composer'] = [payload()]
        inputs = defaultdict(list, {self.capture.MetaField.STEPS: [('sampler', 8)], self.capture.MetaField.SEED: [('sampler', 123)]})
        result = self.capture.Capture.gen_pnginfo_dict(inputs, defaultdict(list), graph(), sampler_node_id='sampler')
        self.assertIn('Scene: Two people in a garden', result['Positive prompt'])
        self.assertIn('Character 1: trigger_a, woman', result['Positive prompt'])
        self.assertIn('Character 2: trigger_b, man', result['Positive prompt'])
        self.assertEqual(result['Negative prompt'], 'blurry, watermark')


if __name__ == '__main__':
    unittest.main()
