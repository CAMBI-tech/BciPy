import unittest

import torch

from bcipy.signal.tests.model.neural_net.rigged_classifier import RiggedClassifier
from bcipy.signal.model.neural_net.probability import update_log_posterior


class TestProbability(unittest.TestCase):
    def test_update_log_posterior__nearly_one_hot_outputs(self):
        batch_size, seq_len, none_class_idx, a_len = 3, 8, 8, 28

        fake_model = RiggedClassifier(
            indices=torch.tensor([0]).repeat(batch_size, 1),
            probs=torch.tensor([0.83]).repeat(batch_size, 1),
            seq_len=seq_len,
            other_seen_probs=torch.tensor([0.01]).repeat(batch_size, 1),
            unseen_class_probs=torch.tensor([0.1]).repeat(batch_size, 1),
            none_class_idx=none_class_idx,
        )

        # presenting letters 0-9, but in different ordering
        presented_seq_idx = torch.stack(
            [
                torch.arange(0, seq_len),
                torch.arange(seq_len // 2, seq_len // 2 + seq_len),
                torch.arange(seq_len, seq_len + seq_len),
            ]
        )

        computed_posterior = update_log_posterior(
            model_log_probs=fake_model(torch.zeros(batch_size, 16, 500)),
            log_prior=(torch.ones(batch_size, a_len) / a_len).log(),  # uniform prior
            presented_seq_idx=presented_seq_idx,
            none_class_idx=none_class_idx,
            alpha=0.8,
        )

        peak = 0.83 / 28 / 0.1  # first item gets most mass from the model
        other_seen = 0.01 / 28 / 0.1  # other seen items get 0.01
        not_seen = 0.005 / 28 / 0.01  # 0.1 is divided across 20 unseen letters (0.005)
        expected_posterior = torch.tensor(
            # first item in batch, model observes letters 0:8
            [
                [peak, *[other_seen] * 7, *[not_seen] * (a_len - seq_len)],
                # next item in batch, model observes letters 4:12
                [
                    *[not_seen] * (seq_len // 2),
                    peak,
                    *[other_seen] * 7,
                    *[not_seen] * (a_len - seq_len - seq_len // 2),
                ],
                # next item in batch, model observes letters 8:16
                [
                    *[not_seen] * seq_len,
                    peak,
                    *[other_seen] * 7,
                    *[not_seen] * (a_len - seq_len - seq_len),
                ],
            ]
        ).log()
        expected_posterior -= expected_posterior.logsumexp(1, keepdim=True)

        self.assertTrue(torch.allclose(expected_posterior, computed_posterior))

    def test_update_log_posterior__exactly_two_hot_outputs(self):
        batch_size, seq_len, none_class_idx, a_len = 3, 8, 8, 28

        fake_model = RiggedClassifier(
            indices=torch.tensor([0, 1]).repeat(batch_size, 1),
            probs=torch.tensor([0.6, 0.3]).repeat(batch_size, 1),
            other_seen_probs=torch.tensor([0.0]).repeat(batch_size, 1),
            unseen_class_probs=torch.tensor([0.1]).repeat(batch_size, 1),
            seq_len=seq_len,
            none_class_idx=none_class_idx,
        )

        # presenting letters 0-9, but in different ordering
        presented_seq_idx = torch.stack([torch.arange(seq_len).roll(-i) for i in range(batch_size)])

        computed_posterior = update_log_posterior(
            model_log_probs=fake_model(torch.zeros(batch_size, 16, 500)),
            log_prior=(torch.ones(batch_size, a_len) / a_len).log(),  # uniform prior
            presented_seq_idx=presented_seq_idx,
            none_class_idx=none_class_idx,
            alpha=0.8,
        )

        # label prior: alpha=0.8, divided among 8 seen letters => 0.1 each, and 0.01 for the 20 other letters
        peak = 0.6 / 28 / 0.1  # first item get most mass
        second = 0.3 / 28 / 0.1  # second item gets some
        other_seen = 0 / 28 / 0.1  # other seen get 0
        not_seen = 0.005 / 28 / 0.01  # rest get 0.1 divided across 20s
        expected_posterior = torch.tensor(
            [
                [peak, second, *[other_seen] * 6, *[not_seen] * (a_len - seq_len)],
                # next item in batch is rolled by 1
                [other_seen, peak, second, *[other_seen] * 5, *[not_seen] * (a_len - seq_len)],
                [*[other_seen] * 2, peak, second, *[other_seen] * 4, *[not_seen] * (a_len - seq_len)],
            ]
        ).log()
        expected_posterior -= expected_posterior.logsumexp(1, keepdim=True)

        self.assertTrue(torch.allclose(expected_posterior, computed_posterior))

    def test_update_log_posterior__nonuniform_prior(self):
        batch_size, seq_len, none_class_idx, a_len = 2, 8, 8, 28

        fake_model = RiggedClassifier(
            indices=torch.tensor([0]).repeat(batch_size, 1),
            probs=torch.tensor([0.83]).repeat(batch_size, 1),
            seq_len=seq_len,
            other_seen_probs=torch.tensor([0.01]).repeat(batch_size, 1),
            unseen_class_probs=torch.tensor([0.1]).repeat(batch_size, 1),
            none_class_idx=none_class_idx,
        )

        # presenting letters 0-9, but in different ordering
        presented_seq_idx = torch.stack([torch.arange(0, seq_len), torch.arange(seq_len, seq_len * 2)])

        # prior
        p0 = 0.6
        p1 = 0.2
        p2 = 0.01
        log_prior = torch.tensor([p0, *[p1] * 7, *[p2] * 20]).expand(batch_size, a_len).log()

        computed_posterior = update_log_posterior(
            model_log_probs=fake_model(torch.zeros(batch_size, 16, 500)),
            log_prior=log_prior,
            presented_seq_idx=presented_seq_idx,
            none_class_idx=none_class_idx,
            alpha=0.8,
        )

        # first item gets most mass from the model
        peak = 0.83 / 0.1
        # other seen items get 0.01
        others = 0.01 / 0.1
        # 0.1 is divided across 20 unseen letters (0.005)
        rest = 0.005 / 0.01
        expected_posterior = (
            torch.tensor(
                # first item in batch, model observes letters 0:8
                [
                    [peak, *[others] * 7, *[rest] * 20],
                    # next item in batch, model observes letters 8:16
                    [rest, *[rest] * 7, peak, *[others] * 7, *[rest] * 12],
                ]
            )
            .log()
            .add(log_prior)
        )

        expected_posterior -= expected_posterior.logsumexp(1, keepdim=True)

        self.assertTrue(torch.allclose(expected_posterior, computed_posterior))
