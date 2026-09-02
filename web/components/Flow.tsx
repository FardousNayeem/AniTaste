"use client";

/**
 * The whole client: one question at a time, then results.
 *
 * The questions are not defined here. They come from the API, which reads
 * them from the same file the scorer uses, so this cannot ask something the
 * model has never heard of.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  fetchQuestions,
  fetchRecommendations,
  type Option,
  type Question,
  type Result,
} from "@/lib/api";

type Phase = "loading" | "asking" | "thinking" | "results" | "broken";

export default function Flow() {
  const [phase, setPhase] = useState<Phase>("loading");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [titles, setTitles] = useState(0);
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState<Record<string, string[]>>({});
  const [picked, setPicked] = useState<string[]>([]);
  const [results, setResults] = useState<Result[]>([]);
  const [seen, setSeen] = useState<number[]>([]);
  const [text, setText] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    fetchQuestions()
      .then((data) => {
        if (!data.available || !data.questions.length) {
          setPhase("broken");
          setError("No model is trained on this server yet.");
          return;
        }
        setQuestions(data.questions);
        setTitles(data.titles);
        setPhase("asking");
      })
      .catch((reason: Error) => {
        setPhase("broken");
        setError(reason.message);
      });
  }, []);

  const question = questions[step];

  const ask = useCallback(
    async (nextAnswers: Record<string, string[]>, freeText: string, exclude: number[] = []) => {
      setPhase("thinking");
      setError("");
      const started = Date.now();
      try {
        const found = await fetchRecommendations({
          answers: nextAnswers,
          text: freeText,
          exclude,
        });
        // An answer that lands instantly reads as canned.
        const elapsed = Date.now() - started;
        if (elapsed < 420) await new Promise((r) => setTimeout(r, 420 - elapsed));
        setResults(found);
        setPhase("results");
      } catch (reason) {
        setError((reason as Error).message);
        setPhase(questions.length ? "asking" : "broken");
      }
    },
    [questions.length]
  );

  const commit = useCallback(
    (values: string[]) => {
      const next = { ...answers, [question.key]: values };
      setAnswers(next);
      setPicked([]);
      if (step + 1 >= questions.length) {
        void ask(next, "");
      } else {
        setStep(step + 1);
      }
    },
    [answers, ask, question, questions.length, step]
  );

  const toggle = (option: Option) => {
    if (!question.multiple) {
      commit([option.value]);
      return;
    }
    setPicked((current) =>
      current.includes(option.value)
        ? current.filter((value) => value !== option.value)
        : current.length < question.maxChoices
          ? [...current, option.value]
          : current
    );
  };

  const restart = () => {
    setAnswers({});
    setPicked([]);
    setStep(0);
    setResults([]);
    setSeen([]);
    setText("");
    setError("");
    setPhase("asking");
  };

  const more = () => {
    // Everything already shown is excluded, so "show me others" genuinely
    // returns others rather than the same ranking again.
    const exclude = [...seen, ...results.map((item) => item.id)];
    setSeen(exclude);
    void ask(answers, text, exclude);
  };

  const submitText = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = text.trim();
    if (!trimmed) return;
    // A typed sentence is a complete request on its own. It goes with
    // whatever has been answered so far and ends the round.
    void ask(answers, trimmed);
  };

  const trained = useMemo(() => titles.toLocaleString("en-GB"), [titles]);

  return (
    <main className="shell">
      <header className="masthead">
        <div className="wordmark">
          <i className="dot" aria-hidden="true" />
          AniTaste
        </div>
        {titles > 0 && (
          <p className="masthead__meta">{trained} titles · Trained on taste</p>
        )}
      </header>

      {phase === "loading" && (
        <div className="stage">
          <span className="thinking" aria-label="Loading">
            <i /><i /><i />
          </span>
        </div>
      )}

      {phase === "broken" && (
        <div className="stage">
          <div className="notice notice--warn">{error}</div>
        </div>
      )}

      {(phase === "asking" || phase === "thinking") && question && (
        <div className="stage">
          <div className="step" key={step}>
            <div className="progress" aria-hidden="true">
              {questions.map((item, index) => (
                <span
                  key={item.key}
                  className={
                    index < step ? "is-done" : index === step ? "is-current" : ""
                  }
                />
              ))}
            </div>

            <p className="eyebrow">
              Question {step + 1} of {questions.length}
            </p>
            <h1 className="prompt">{question.prompt}</h1>
            {question.hint && <p className="hint">{question.hint}</p>}

            {phase === "thinking" ? (
              <span className="thinking" aria-label="Thinking">
                <i /><i /><i />
              </span>
            ) : (
              <>
                <div className="options">
                  {question.options.map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      className="chip"
                      aria-pressed={picked.includes(option.value)}
                      onClick={() => toggle(option)}
                    >
                      {picked.includes(option.value) && (
                        <span className="chip__tick" aria-hidden="true">✓</span>
                      )}
                      {option.label}
                    </button>
                  ))}
                </div>

                {question.multiple && (
                  <div className="row">
                    <button
                      type="button"
                      className="btn"
                      disabled={!picked.length && !question.optional}
                      onClick={() => commit(picked)}
                    >
                      {picked.length
                        ? `That's me (${picked.length})`
                        : question.optional
                          ? "Nothing to avoid"
                          : "Pick atleast one"}
                    </button>
                    {step > 0 && (
                      <button type="button" className="btn btn--quiet" onClick={restart}>
                        Start over
                      </button>
                    )}
                  </div>
                )}
              </>
            )}

            {error && <p className="notice notice--warn" style={{ marginTop: "1.5rem" }}>{error}</p>}
          </div>
            
          <form className="compose" onSubmit={submitText}>
            <label className="compose__label" htmlFor="wish">
              In a hurry? Describe it, or name an anime you liked.
            </label>
            <div className="compose__row">
              <input
                id="wish"
                className="input"
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="something like Cowboy Bebop, but funnier"
                maxLength={300}
                autoComplete="off"
              />
              <button type="submit" className="btn" disabled={!text.trim()}>
                Ask
              </button>
            </div>
          </form>
        </div>
      )}

      {phase === "results" && (
        <div className="results">
          <div className="results__head">
            <h1 className="results__title">
              {results.length && results[0].match >= 25
                ? "Here is what I would put on."
                : "Nothing is a close match, but these are the nearest."}
            </h1>
            <div className="row">
              <button type="button" className="btn btn--quiet" onClick={more}>
                Show me others
              </button>
              <button type="button" className="btn" onClick={restart}>
                Start over
              </button>
            </div>
          </div>

          <div className="grid">
            {results.map((item) => (
              <a
                key={item.id}
                className="card"
                href={item.url}
                target="_blank"
                rel="noreferrer noopener"
                title={`Search the web for ${item.name}`}
              >
                {item.poster ? (
                  <img className="card__art" src={item.poster} alt="" loading="lazy" />
                ) : (
                  <span className="card__art" aria-hidden="true" />
                )}
                <div className="card__body">
                  <span className="card__name">{item.name}</span>
                  <span className="card__meta">
                    {item.year && <span>{item.year}</span>}
                    {item.episodes && (
                      <span>
                        {item.episodes} ep{item.episodes === 1 ? "" : "s"}
                      </span>
                    )}
                    {item.score && <span>{item.score}%</span>}
                  </span>
                  {item.reasons.length > 0 && (
                    <span className="tags">
                      {item.reasons.map((reason) => (
                        <span className="tag" key={reason}>
                          {reason}
                        </span>
                      ))}
                    </span>
                  )}
                  {item.neighbours.length > 0 && (
                    <span className="card__note">
                      Fans also watch {item.neighbours.join(" and ")}.
                    </span>
                  )}
                  <span className="match">
                    <span className="bar">
                      <span style={{ width: `${Math.max(3, item.match)}%` }} />
                    </span>
                    <span className="match__value">{item.match}%</span>
                  </span>
                </div>
              </a>
            ))}
          </div>

          <p className="footnote">
            Ranked by a model trained on {trained} titles and the community
            recommendation graph of what people who liked.
          </p>
        </div>
      )}
    </main>
  );
}
