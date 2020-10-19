import unittest

from collections import Counter
from bcipy.helpers.language_model import norm_domain, sym_appended, \
    equally_probable, histogram


class TestLanguageModelRelated(unittest.TestCase):
    def test_norm_domain(self):
        """Test conversion from negative log likelihood to prob."""
        letters = [('S', 0.25179717717251715), ('U', 1.656395297172517),
                   ('O', 4.719718077172517), ('M', 4.824790097172517),
                   ('W', 4.846891977172517), ('T', 6.100397207172517),
                   ('P', 6.8986402471725174), ('R', 7.081149197172517),
                   ('L', 7.108869167172517), ('N', 7.508945087172517),
                   ('_', 8.251687627172517), ('C', 8.670805547172517),
                   ('E', 8.820671657172516), ('B', 8.838797187172517),
                   ('A', 9.040823557172518), ('D', 9.134643177172517),
                   ('H', 9.134643177172517), ('G', 9.193730927172517),
                   ('F', 9.265835427172517), ('V', 9.374314927172517),
                   ('K', 9.569215427172518), ('I', 9.648203427172517),
                   ('Y', 10.942930827172518), ('J', 11.299606927172517),
                   ('X', 12.329225127172517), ('Z', 12.329227027172518),
                   ('Q', 13.245515427172517)]
        expected = [
            ('S', 0.7774023970322453), ('U', 0.19082561142814908),
            ('O', 0.008917692295108082), ('M', 0.008028238843581626),
            ('W', 0.00785274617485694), ('T', 0.0022419770132497793),
            ('P', 0.0010091567002187994), ('R', 0.0008408063406892647),
            ('L', 0.0008178192862913141), ('N', 0.0005481590438212282),
            ('_', 0.0002608180220954618), ('C', 0.00017152088039886618),
            ('E', 0.0001476491573050645), ('B', 0.0001449970461439091),
            ('A', 0.0001184732267906119), ('D', 0.00010786359081437584),
            ('H', 0.00010786359081437584), ('G', 0.00010167481484983796),
            ('F', 9.460167015257232e-05), ('V', 8.487636182290733e-05),
            ('K', 6.984616150492476e-05), ('I', 6.454141629213861e-05),
            ('Y', 1.7682575696535268e-05), ('J', 1.2377788678084351e-05),
            ('X', 4.420644194323101e-06), ('Z', 4.420635795107107e-06),
            ('Q', 1.7682584413941958e-06)
        ]
        for i, pair in enumerate(norm_domain(letters)):
            self.assertEqual(expected[i][0], pair[0])
            self.assertAlmostEqual(expected[i][1], pair[1])

    def test_insert_sym_with_zero_prob(self):
        """Test insertion of an additional symbol (with zero probability) to a
        normalized list of symbols."""

        syms = [
            ('S', 0.21999999999999953), ('U', 0.03), ('O', 0.03), ('M', 0.03),
            ('W', 0.03), ('T', 0.03), ('P', 0.03), ('R', 0.03), ('L', 0.03),
            ('N', 0.03), ('_', 0.03), ('C', 0.03), ('E', 0.03), ('B', 0.03),
            ('A', 0.03), ('D', 0.03), ('H', 0.03), ('G', 0.03), ('F', 0.03),
            ('V', 0.03), ('K', 0.03), ('I', 0.03), ('Y', 0.03), ('J', 0.03),
            ('X', 0.03), ('Z', 0.03), ('Q', 0.03)
        ]

        self.assertEqual(1.0, sum([prob for _, prob in syms]))

        new_list = sym_appended(syms, ('<', 0.0))
        self.assertEqual(len(syms) + 1, len(new_list))
        self.assertEqual(1.0, sum([prob for _, prob in new_list]))

        for pair in syms:
            self.assertTrue(pair in new_list)
        self.assertTrue('<' in dict(new_list))

    def test_insert_sym_with_non_zero_prob(self):
        """Test insertion of an additional symbol to a normalized list of
        symbols with a non-zero probability."""

        syms = [('A', 0.25), ('B', 0.25), ('C', 0.25), ('D', 0.25)]

        self.assertEqual(1.0, sum([prob for _, prob in syms]))

        new_sym = ('<', 0.2)

        new_list = sym_appended(syms, new_sym)
        self.assertEqual(len(syms) + 1, len(new_list))
        self.assertAlmostEqual(1.0, sum([prob for _, prob in new_list]))

        new_list_dict = dict(new_list)
        prev_list_dict = dict(syms)
        for s, _ in syms:
            self.assertTrue(s in new_list_dict)
            self.assertTrue(new_list_dict[s] < prev_list_dict[s])
            self.assertEqual(0.2, new_list_dict[s])
        self.assertTrue(new_sym[0] in new_list_dict)
        self.assertEqual(new_sym[1], new_list_dict[new_sym[0]])

    def test_insert_sym_when_already_exists(self):
        """Test insertion of an additional symbol to a normalized list of
        symbols."""

        syms = [('A', 0.25), ('B', 0.25), ('C', 0.25), ('D', 0.25)]
        self.assertEqual(1.0, sum([prob for _, prob in syms]))

        new_list = sym_appended(syms, ('D', 0.25))
        self.assertEqual(syms, new_list, "Value already present")

        new_list = sym_appended(syms, ('D', 0.2))
        self.assertEqual(
            syms, new_list, msg="Changing the probability does not matter")

    def test_equally_probable(self):
        """Test generation of equally probable values."""

        # no overrides
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp)
        self.assertEqual(len(alp), len(probs))
        for prob in probs:
            self.assertEqual(0.25, prob)

        # test with override
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'A': 0.4})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.4)
        self.assertAlmostEqual(probs[1], 0.2)
        self.assertAlmostEqual(probs[2], 0.2)
        self.assertAlmostEqual(probs[3], 0.2)

        # test with 0.0 override
        alp = ['A', 'B', 'C', 'D', 'E']
        probs = equally_probable(alp, {'E': 0.0})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.25)
        self.assertAlmostEqual(probs[1], 0.25)
        self.assertAlmostEqual(probs[2], 0.25)
        self.assertAlmostEqual(probs[3], 0.25)
        self.assertAlmostEqual(probs[4], 0.0)

        # test with multiple overrides
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'B': 0.2, 'D': 0.3})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        self.assertEqual(probs[0], 0.25)
        self.assertAlmostEqual(probs[1], 0.2)
        self.assertAlmostEqual(probs[2], 0.25)
        self.assertAlmostEqual(probs[3], 0.3)

        # test with override that's not in the alphabet
        alp = ['A', 'B', 'C', 'D']
        probs = equally_probable(alp, {'F': 0.4})
        self.assertEqual(len(alp), len(probs))
        self.assertAlmostEqual(1.0, sum(probs))
        for prob in probs:
            self.assertEqual(0.25, prob)

    def test_small_probs(self):
        """When very small values are returned from the LM, inserting a letter
        should still result in all positive values"""
        probs = [('_', 0.8137718053286306), ('R', 0.04917114015944412),
                 ('Y', 0.04375449276342169), ('I', 0.03125895356629575),
                 ('M', 0.023673042636520744), ('S', 0.018415576386909806),
                 ('N', 0.014673750822550981), ('O', 0.003311888694636908),
                 ('A', 0.0015325727808248953), ('E', 0.00020663161460758318),
                 ('F', 0.0001271103705188304), ('L', 7.17785373200501e-05),
                 ('T', 1.9445808941289728e-05), ('V', 8.947029414950125e-06),
                 ('D', 1.3287314209822164e-06), ('W', 5.781802939202195e-07),
                 ('C', 4.956713702874677e-07), ('U', 1.3950738615826266e-07),
                 ('J', 6.925441151532328e-08), ('H', 6.067034614011934e-08),
                 ('B', 4.83364892878174e-08), ('K', 4.424005264637171e-08),
                 ('P', 3.319195566216423e-08), ('Z', 2.9048107858874575e-08),
                 ('X', 1.904356496190087e-08), ('Q', 1.0477572781302604e-08),
                 ('G', 7.146978265955833e-09)]

        syms = sym_appended(probs, ('<', 0.05))
        self.assertAlmostEqual(1.0, sum([sym[1] for sym in syms]))
        for sym in syms:
            self.assertTrue(sym[1] >= 0)

    def test_histogram(self):
        """Test histogram visualization."""

        probs = [('_', 0.8137718053286306), ('R', 0.04917114015944412),
                 ('Y', 0.04375449276342169), ('I', 0.03125895356629575),
                 ('M', 0.023673042636520744), ('S', 0.018415576386909806),
                 ('N', 0.014673750822550981), ('O', 0.003311888694636908),
                 ('A', 0.0015325727808248953), ('E', 0.00020663161460758318),
                 ('F', 0.0001271103705188304), ('L', 7.17785373200501e-05),
                 ('T', 1.9445808941289728e-05), ('V', 8.947029414950125e-06),
                 ('D', 1.3287314209822164e-06), ('W', 5.781802939202195e-07),
                 ('C', 4.956713702874677e-07), ('U', 1.3950738615826266e-07),
                 ('J', 6.925441151532328e-08), ('H', 6.067034614011934e-08),
                 ('B', 4.83364892878174e-08), ('K', 4.424005264637171e-08),
                 ('P', 3.319195566216423e-08), ('Z', 2.9048107858874575e-08),
                 ('X', 1.904356496190087e-08), ('Q', 1.0477572781302604e-08),
                 ('G', 7.146978265955833e-09)]

        hist = histogram(probs)
        lines = hist.split("\n")
        self.assertEqual(len(lines), len(probs))
        self.assertTrue(
            all(lines[i] <= lines[i + 1] for i in range(len(lines) - 1)),
            "Should be sorted")

        self.assertTrue(lines[-1].startswith('_'))
        c = Counter(lines[-1])
        self.assertEqual(81, c['*'],
                         "Should be 81 stars, indicating the percentage")

        total_stars = Counter(hist)['*']
        self.assertTrue(
            total_stars <= 100 and total_stars >= 95,
            "Total number of stars should almost equal 100; exact value depends on rounding"
        )
