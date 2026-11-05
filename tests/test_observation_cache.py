"""CPU-only live-schema selection and request forwarding, not GPU acceptance."""
import unittest
from unittest.mock import patch
from rtx5090 import eager


class CacheIsolation(unittest.IsolatedAsyncioTestCase):
    def api(self, field):
        return {'paths': {'/v1/completions': {'post': {'requestBody': {'content': {
            'application/json': {'schema': {'$ref': '#/components/schemas/Completion'}}}}}}},
            'components': {'schemas': {'Completion': {'properties': field}}}}

    async def isolate(self, api, request):
        with patch.object(eager.service, 'prefix_policy', return_value=True), \
             patch.object(eager.client_api, 'http', return_value=api):
            return await eager.isolate_observation_cache({'base_url': 'test'}, request)

    async def test_salt_unique_and_not_claimed_verified(self):
        api = self.api({'cache_salt': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}})
        a, b = {}, {}
        result = await self.isolate(api, a)
        await self.isolate(api, b)
        self.assertNotEqual(a['cache_salt'], b['cache_salt'])
        self.assertFalse(result['miss_verified'])
        self.assertEqual(result['method'], 'unique_request_cache_salt')

    async def test_unadvertised_or_wrong_type_rejected(self):
        for field in ({}, {'cache_salt': {'type': 'integer'}}):
            with self.assertRaisesRegex(RuntimeError, 'advertised string'):
                await self.isolate(self.api(field), {})

    async def test_recursive_schema_rejected(self):
        api = self.api({})
        api['components']['schemas']['Completion'] = {'$ref': '#/components/schemas/Completion'}
        with self.assertRaisesRegex(ValueError, 'resolve'):
            await self.isolate(api, {})

    async def test_existing_reset_kept_and_failure_not_bypassed(self):
        api = self.api({'cache_salt': {'type': 'string'}})
        api['paths']['/reset_prefix_cache'] = {'post': {}}
        with patch.object(eager.service, 'prefix_policy', return_value=True), \
             patch.object(eager.client_api, 'http', side_effect=[api, False]):
            with self.assertRaisesRegex(RuntimeError, 'reset failed'):
                await eager.isolate_observation_cache({'base_url': 'test'}, {})

    async def test_paired_window_gate_unchanged(self):
        api = self.api({'cache_salt': {'type': 'string'}})
        with patch.object(eager.service, 'prefix_policy', return_value=True), \
             patch.object(eager.client_api, 'http', return_value=api):
            with self.assertRaisesRegex(RuntimeError, 'Paired prompts'):
                await eager.clear_cache({'base_url': 'test'})
