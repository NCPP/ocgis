from ocgis.calc.gen_xclim_api import create_xclim_defn, get_xclim_signature
from ocgis.test.base import TestBase
from ocgis.util.addict import Dict


class Test(TestBase): #tdk:rename

    def test_create_xclim_defn(self):
        apidef = create_xclim_defn()
        self.assertIsInstance(apidef, Dict)

    def test_get_xclim_signature(self):
        apidef = create_xclim_defn()
        key = "cold_spell_duration_index"
        sig = get_xclim_signature(key, apidef)
        self.assertIsNone(sig["args"]["tn10"])
