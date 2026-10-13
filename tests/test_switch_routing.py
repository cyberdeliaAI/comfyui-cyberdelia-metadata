"""Core lazy switches must not leak inactive prompts or LoRAs into metadata."""
import importlib.util
import asyncio
import sys
import types
import unittest
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_switch_capture_fixture', ROOT/'tests/test_negpip_prompt_routing.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def graph(enabled, linked=True, negative_socket=False):
    prompt = {
        'toggle': {'class_type': 'PrimitiveBoolean', 'inputs': {'value': enabled}},
        'normal': {'class_type': 'ZImageNegPipPrompt', 'inputs': {
            'positive': 'normal scene', 'negative': 'normal negative'}},
        'composer': {'class_type': 'Krea2TwoCharacterComposer', 'inputs': {
            'base_prompt': 'regional scene', 'negative_prompt': 'regional negative',
            'character_1_prompt': 'a woman', 'character_2_prompt': 'a man',
            'character_1_lora': 'woman.safetensors', 'character_1_strength': 1,
            'character_2_lora': 'man.safetensors', 'character_2_strength': .8}},
        'sampler': {'class_type': 'CyberKreaSampler', 'inputs': {
            'positive': ['positive_switch', 0], 'model': ['model_switch', 0],
            'seed': 1, 'steps': 8}},
        'context': {'class_type': 'Context Big (rgthree)', 'inputs': {
            'positive': ['positive_switch', 0], 'negative': ['negative_switch', 0]}},
        'save': {'class_type': 'CyberdeliaSaveImageWithMetaData', 'inputs': {
            'images': ['sampler', 0], 'context': ['context', 0]}},
    }
    for role, off_slot, on_slot in [('model', 0, 0), ('positive', 1, 1), ('negative', 2, 4)]:
        prompt[role+'_switch'] = {'class_type': 'ComfySwitchNode', 'inputs': {
            'switch': ['toggle', 0] if linked else enabled,
            'on_false': ['normal', off_slot], 'on_true': ['composer', on_slot]}}
    if negative_socket:
        prompt['sampler']['inputs']['negative'] = ['negative_switch', 0]
    return prompt


class SwitchRoutingTests(unittest.TestCase):
    def setUp(self):
        self.capture = fixture._load_capture()
        self.capture._clear_resolved_texts()
        package = self.capture.__package__
        fixture._load_module(package+'.defs.samplers', ROOT/'modules/defs/samplers.py')
        self.trace = fixture._load_module(package+'.trace', ROOT/'modules/trace.py').Trace

    def test_normal_and_composer_roles_follow_same_boolean(self):
        for enabled in (False, True):
            for linked in (False, True):
                for negative_socket in (False, True):
                    with self.subTest(enabled=enabled, linked=linked, negative_socket=negative_socket):
                        prompt = graph(enabled, linked, negative_socket)
                        positive, negative = self.capture._find_prompt_texts(prompt, None, sampler_node_id='sampler')
                        if enabled:
                            self.assertIn('Scene: regional scene', positive)
                            self.assertIn('Character 2: a man', positive)
                            self.assertNotIn('normal scene', positive)
                            self.assertEqual(negative, 'regional negative')
                        else:
                            self.assertEqual(positive, 'normal scene')
                            self.assertEqual(negative, 'normal negative')

    def test_trace_and_prompt_fallback_exclude_inactive_resource_branch(self):
        for enabled in (False, True):
            prompt = graph(enabled)
            active, inactive = ('composer', 'normal') if enabled else ('normal', 'composer')
            traced = self.trace.trace('save', prompt)
            self.assertIn(active, traced)
            self.assertNotIn(inactive, traced)
            upstream = {nid for _, nid, _ in self.capture._walk_upstream_nodes('sampler', prompt)}
            self.assertIn(active, upstream)
            self.assertNotIn(inactive, upstream)
            self.assertIn('toggle', traced)

    def test_connected_context_uses_active_prompt_pair(self):
        for enabled in (False, True):
            positive, negative = self.capture.Capture.resolve_context_prompts(graph(enabled), 'context')
            self.assertEqual(negative, 'regional negative' if enabled else 'normal negative')
            self.assertIn('regional scene' if enabled else 'normal scene', positive)

    def test_selected_empty_prompt_is_not_replaced_by_inactive_text(self):
        prompt = graph(True, negative_socket=True)
        prompt['composer']['inputs']['negative_prompt'] = ''
        self.assertEqual(self.capture._follow_conditioning_to_clip_text(['negative_switch', 0], prompt, None), '')

    def test_text_switch_selects_only_current_branch_without_output_cache(self):
        for enabled in (False, True):
            prompt = graph(enabled)
            prompt['text'] = {'class_type': 'ComfySwitchNode', 'inputs': {
                'switch': ['toggle', 0], 'on_false': 'normal text', 'on_true': 'regional text'}}
            self.assertEqual(self.capture._resolve_text_from_graph(['text', 0], prompt, None),
                             'regional text' if enabled else 'normal text')

    def test_switching_off_ignores_cached_composer_in_saved_metadata(self):
        capture = self.capture
        capture.Trace = self.trace
        package = capture.__package__
        helper = sys.modules[package+'.composer_metadata']
        extension = fixture._load_module(package+'.defs.ext.krea2_composer',
                                         ROOT/'modules/defs/ext/krea2_composer.py')
        extension.calc_lora_hash = lambda name: 'hash-'+name
        capture.CAPTURE_FIELD_LIST.update(extension.CAPTURE_FIELD_LIST)
        meta = capture.MetaField
        capture.CAPTURE_FIELD_LIST['CyberKreaSampler'] = {
            meta.STEPS: {'field_name': 'steps'}, meta.SEED: {'field_name': 'seed'}}
        for kind in ('CyberKreaSampler', 'Krea2TwoCharacterComposer'):
            capture.NODE_CLASS_MAPPINGS[kind] = type(kind, (), {})
        normal_prompt = 'A woman at an outdoor music festival at sunset, warm film colors.'
        regional = {'version': 1, 'scene_prompt': normal_prompt + ' Two distinct characters in one coherent scene.',
                    'character_prompts': ['a woman wearing a red coat', 'a man wearing a blue jacket'],
                    'negative_prompt': 'regional negative',
                    'loras': [[{'name': 'woman.safetensors', 'strength': 1}],
                              [{'name': 'man.safetensors', 'strength': .8}]]}
        cached_composer = [[], [[['tensor stand-in', {helper.METADATA_KEY: regional}]]],
                           [], [], [], ['Scene: diagnostic with Character 1 and Character 2']]
        class Cache:
            cache = {}
            async def get(self, node_id):
                return types.SimpleNamespace(outputs=cached_composer) if node_id == 'composer' else None
        hook = sys.modules[package+'.hook']
        hook.current_extra_data = {}
        hook.prompt_executer = types.SimpleNamespace(caches=types.SimpleNamespace(outputs=Cache()))
        execution = types.ModuleType('execution')
        async def get_input_data(inputs, *args):
            return ({key: [value] for key, value in inputs.items()}, {}, {})
        execution.get_input_data = get_input_data
        sys.modules['execution'] = execution
        graph_module = types.ModuleType('comfy_execution.graph')
        graph_module.DynamicPrompt = lambda prompt: prompt
        sys.modules['comfy_execution'] = types.ModuleType('comfy_execution')
        sys.modules[graph_module.__name__] = graph_module

        # Preserve the cache across ON -> OFF -> OFF, just as cached execution does.
        for enabled in (True, False, False):
            prompt = graph(enabled)
            prompt['normal']['inputs']['positive'] = normal_prompt
            hook.current_prompt = prompt
            hook.current_save_image_node_id = 'save'
            self.trace._trace_cache.clear()
            captured = asyncio.run(capture.Capture.get_inputs())
            overrides = capture.Capture.resolve_context_prompts(prompt, 'context')
            # A bypassed rgthree context can leave the save node without an
            # explicit context connection. Both save routes must stay correct.
            for prompt_overrides in (None, overrides):
                result = capture.Capture.gen_pnginfo_dict(
                    captured, defaultdict(list), prompt, sampler_node_id='sampler',
                    active_trace_tree=self.trace.trace('save', prompt), prompt_overrides=prompt_overrides)
                if enabled:
                    self.assertIn('Character 2:', result['Positive prompt'])
                    self.assertIn('composer', capture._resolved_composer_metadata)
                    continue
                self.assertEqual(result['Positive prompt'], normal_prompt)
                self.assertEqual(result['Negative prompt'], 'normal negative')
                self.assertNotIn('composer', capture._resolved_composer_metadata)
                self.assertFalse(captured.get(meta.LORA_MODEL_NAME))
                self.assertFalse(any(key.startswith('Krea character') for key in result))

    def test_unknown_boolean_and_cycles_do_not_crash_or_guess(self):
        helper = sys.modules[self.capture.__package__+'.switching']
        prompt = graph(True)
        prompt['toggle']['inputs']['value'] = ['toggle', 0]
        self.assertIsNone(helper.selected_switch_input(prompt['model_switch'], prompt))
        for value in (None, 'false', 1, [[False], 0], ['toggle', False], ['toggle', 0.0]):
            self.assertIsNone(helper.resolve_boolean(value, prompt))


if __name__ == '__main__':
    unittest.main()
