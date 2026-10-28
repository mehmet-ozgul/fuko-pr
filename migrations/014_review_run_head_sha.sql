-- The commit each review run actually reviewed.
--
-- A run row could say which PR, model and seat ran, but not on which head, so
-- nothing outside the runner could tell whether a head was already covered: a
-- scheduler that reviews new pushes re-ran every head it saw. The agentic
-- backend knows the commit it checked out and reports it; PR-Agent never
-- resolves one.
--
-- NULLABLE and not backfilled, for the reason 008 gives: older rows and every
-- pr-agent row have an unknown head, and any value written for them would be a
-- guess presented as a receipt.
--
-- Idempotent, like every migration here: they are re-run on each pool creation.
ALTER TABLE review_runs ADD COLUMN IF NOT EXISTS head_sha TEXT;

CREATE INDEX IF NOT EXISTS review_runs_repo_pr_idx ON review_runs (repo, pr, started_at DESC);
