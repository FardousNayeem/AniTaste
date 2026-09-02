// The one place that knows the API's shape. Requests go to the same origin;
// next.config.mjs proxies /api to the Python service.

export type Option = { value: string; label: string; icon?: string };

export type Question = {
  key: string;
  prompt: string;
  hint: string;
  multiple: boolean;
  maxChoices: number;
  optional: boolean;
  options: Option[];
};

export type Result = {
  id: number;
  name: string;
  year: number | null;
  episodes: number | null;
  poster: string;
  colour: string;
  score: number | null;
  match: number;
  reasons: string[];
  neighbours: string[];
  // Where a click goes: a web search for the title.
  url: string;
  // The entry the model learned it from, kept for anyone who wants it.
  anilistUrl: string;
};

export type Questionnaire = {
  available: boolean;
  questions: Question[];
  titles: number;
  trainedAt: string;
};

export async function fetchQuestions(): Promise<Questionnaire> {
  const response = await fetch("/api/questions");
  if (!response.ok) throw new Error("The recommender is not answering.");
  return response.json();
}

export async function fetchRecommendations(body: {
  answers: Record<string, string[]>;
  text: string;
  exclude?: number[];
  limit?: number;
}): Promise<Result[]> {
  const response = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit: 6, exclude: [], ...body }),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    // FastAPI puts the reason in `detail`, and those messages are written to
    // be shown to a person.
    throw new Error(payload?.detail ?? "Something went wrong. Try again?");
  }
  return payload.results as Result[];
}
