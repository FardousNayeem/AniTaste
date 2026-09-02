"""The library's public behaviour, against the shipped model."""

from __future__ import annotations

import unittest

from anitaste.model import ModelUnavailable, load_model
from anitaste.recommend import Ask, InvalidAnswer, questions, recommend, status

try:
    load_model()
    HAS_MODEL = True
except ModelUnavailable:
    HAS_MODEL = False


@unittest.skipUnless(HAS_MODEL, "no trained model in data/")
class QuestionnaireTests(unittest.TestCase):
    def test_the_questionnaire_is_served_with_options(self):
        payload = questions()
        self.assertTrue(payload)
        self.assertTrue(all(question["options"] for question in payload))

    def test_status_reports_the_metrics(self):
        report = status()
        self.assertTrue(report["available"])
        self.assertGreater(report["titles"], 1000)
        self.assertIn("recall_at_10", report["metrics"])


@unittest.skipUnless(HAS_MODEL, "no trained model in data/")
class ValidationTests(unittest.TestCase):
    def test_an_unknown_question_is_rejected(self):
        with self.assertRaises(InvalidAnswer):
            recommend(Ask(answers={"favourite-colour": ["blue"]}))

    def test_an_unknown_option_is_rejected(self):
        with self.assertRaises(InvalidAnswer):
            recommend(Ask(answers={"mood": ["sepia"]}))

    def test_too_many_choices_are_rejected(self):
        with self.assertRaises(InvalidAnswer):
            recommend(Ask(answers={"mood": ["dark", "funny", "action"]}))

    def test_an_empty_ask_is_rejected(self):
        with self.assertRaises(InvalidAnswer):
            recommend(Ask())

    def test_unreadable_text_says_so(self):
        with self.assertRaises(InvalidAnswer) as caught:
            recommend(Ask(text="zzzzqx wibblefrotz"))
        self.assertIn("model knows", str(caught.exception))


@unittest.skipUnless(HAS_MODEL, "no trained model in data/")
class RecommendationTests(unittest.TestCase):
    def test_a_mood_returns_ranked_results(self):
        results = recommend(Ask(answers={"mood": ["dark"]}, limit=5))
        self.assertEqual(len(results), 5)
        self.assertGreaterEqual(results[0].match, results[-1].match)

    def test_every_result_is_presentable(self):
        for item in recommend(Ask(answers={"mood": ["funny"]}, limit=3)):
            self.assertTrue(item.name)
            self.assertTrue(item.url.startswith("https://duckduckgo.com/?q="))
            self.assertTrue(item.anilist_url.startswith("https://anilist.co/anime/"))
            self.assertGreaterEqual(item.match, 0)
            self.assertLessEqual(item.match, 100)

    def test_the_mood_changes_the_answer(self):
        dark = {item.anilist_id for item in recommend(Ask(answers={"mood": ["dark"]}))}
        funny = {item.anilist_id for item in recommend(Ask(answers={"mood": ["funny"]}))}
        self.assertFalse(dark & funny)

    def test_a_click_searches_for_the_title(self):
        """Names with punctuation must survive into the query string."""
        from urllib.parse import parse_qs, urlparse

        for item in recommend(Ask(text="something like Cowboy Bebop", limit=4)):
            query = parse_qs(urlparse(item.url).query)
            self.assertEqual(query["q"], [item.name])

    def test_a_named_title_is_not_recommended_back(self):
        """"Something like X" means something other than X."""
        results = recommend(Ask(text="something like Cowboy Bebop", limit=6))
        self.assertNotIn("Cowboy Bebop", [item.name for item in results])

    def test_a_named_title_steers_the_neighbourhood(self):
        names = [item.name for item in recommend(Ask(text="something like Cowboy Bebop", limit=6))]
        self.assertTrue(any("Trigun" in name for name in names), names)

    def test_exclude_returns_different_titles(self):
        first = recommend(Ask(answers={"mood": ["dark"]}, limit=3))
        again = recommend(
            Ask(answers={"mood": ["dark"]}, limit=3, exclude=frozenset(i.anilist_id for i in first))
        )
        self.assertFalse({i.anilist_id for i in first} & {i.anilist_id for i in again})

    def test_a_refusal_is_honoured(self):
        results = recommend(Ask(text="something funny, not too long", limit=6))
        # Nothing enormous should survive "not too long".
        self.assertTrue(all((item.episodes or 0) <= 60 for item in results))


if __name__ == "__main__":
    unittest.main()
