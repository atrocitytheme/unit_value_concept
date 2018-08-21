"""basic function tests
"""
import unittest
import Unit_Value.words_handler

class Basic_Test(unittest.TestCase):
    """test the basic recognition effect
    """
    def test_direct_unit_extractor(self):
        """test the effect of unit recognition
        """
        expect = ['km/h', 'kg/24h', 'L/h', 'L']
        input_values = ['12323km/h', '41231kg/24h', '123223L/h', '9.1L']

        for i, j in zip(expect, input_values):
            comp = Unit_Value.words_handler.process_unit(j)
            self.assertEqual(i, comp, "expected: {} \n but got: {}".format(i, comp))

    def test_unit_similarity_detect(self):
        expect = ['12323', '41231', '123223', '9.1']
        input_values = ['12323km/h', '41231kg/24h', '123223L/h', '9.1L']
        for i, j in zip(expect, input_values):
            value_extracted = Unit_Value.words_handler.process_value(i)
            self.assertEqual(i, value_extracted, "expected: {} \n but got: {}".format(i, value_extracted))


if __name__ == '__main__':
    unittest.main()
