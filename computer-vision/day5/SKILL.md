---
name: oz-assignment-submit
description: Automates submitting an OZ 코딩스쿨 bootcamp day-assignment end to end — checking the Notion submission status, finding the assignment spec, doing the real work, pushing it to GitHub, embedding the link back in Notion, and marking the day complete. Use this whenever 김종민 says something like "<트랙명> N일차 진행하자" / "N일차 이어서 하자" / "딥러닝 3일차 하자" (a track name plus a day number plus "진행/이어서/시작하자") for any OZ AI 헬스케어 bootcamp track — Computer Vision, 딥러닝, 데이터분석, NLP, 머신러닝 심화, etc. Also use it when asked to check what's left across days, since it knows the per-day submission-status pattern.
---

# OZ Assignment Submit

## Why this exists

Every OZ bootcamp track (Computer Vision, 딥러닝, 데이터분석, NLP, ...) uses the same Notion
structure: a per-day 과제 제출란 table with one row per student, a 상태 property
(시작 전/진행 중/완료), and a "필수 과제" block to attach the deliverable. The mechanical
parts of finishing a day — reading Notion, finding the real assignment spec when it's a
vague "자유과제", pushing to GitHub, embedding the link, flipping 상태 to 완료 — are identical
every time. What changes is only the actual technical work (train a classifier, run object
detection, build a report, ...). This skill owns the mechanical wrapper so the actual work
gets full attention instead of being interleaved with UI fumbling.

## Non-negotiable rule: never fabricate results

Every number that goes into a report or gets marked 완료 must come from something that
actually ran — a real dataset, a real training run, a real evaluation. If a genuinely
required dataset needs a Kaggle (or similar) login, stop and ask 김종민 to download it
himself (see "Data acquisition boundary" below) rather than guessing plausible numbers or
skipping the run. A wrong number that looks plausible is worse than an honest "still
running" — it silently corrupts a grade record.

## Step 1 — Check the day's status

1. Open the track's CV/DL/... roadmap page in Notion (or navigate directly if you already
   have the URL from earlier in the session), find the row for the requested day, open it.
2. Find 김종민's row in that day's 과제 제출란 table and read 상태.
   - If already 완료: report that and stop — don't redo finished work.
   - If 시작 전 or 진행 중: continue to Step 2.

Notion's UI here is a bit fragile for automation: clicking a table-row name directly often
does nothing useful — you need to hover to reveal an "열기" button and click that (or use
`find`/`read_page` to get the row's ref, click it, then read the resulting URL from
`tabs_context_mcp` and `navigate` there directly rather than trusting a side-panel to open).
If a click doesn't navigate, screenshot before clicking again — don't just repeat the same
click blindly.

## Step 2 — Find the real assignment spec

Free-form "자유과제" days rarely have the full spec on the surface page. In priority order:

1. **External lecture material first.** Look for a mention/link (often styled like
   "ysshin 오즈코딩스쿨 6기 CV" or a 📍-prefixed page link) that leads to a
   `dainus.notion.site/...` page. These carry the actual dataset name, model, metrics, and
   step-by-step lecture flow — read the whole thing before assuming you understand the task.
2. **If that's missing or insufficient, reverse-engineer from a completed classmate.** Open
   the day's 과제 제출란 table, find any row with 상태 = 완료, open their submission. Use it
   only to recover the *spec* (dataset, model type, metrics, deliverable format) — never
   copy their code or numbers. If their deliverable is an embedded HTML report, it may
   render blank at first; click the block's "..." menu or the expand icon and wait a moment
   — it usually loads. If it's an uploaded `.html` file, clicking it can trigger a download
   that closes the tab almost instantly; don't fight that — if a copy lands in Downloads/ as
   a side effect, reading the already-downloaded file is fine, but don't deliberately force
   a new download without telling 김종민 first (see "Data acquisition boundary" for the
   general download-permission rule).
3. **If the spec still doesn't pin down a dataset** (a genuinely open "find your own data"
   day), ask 김종민 for a domain preference before picking one — don't silently choose for
   him on something this open-ended. Once a domain is chosen, picking the specific dataset
   yourself is fine.

## Step 3 — Do the real work

This is the part that isn't scriptable — follow whatever the spec actually asks for. Some
recurring patterns worth knowing about going in:

- **Prefer GPU when available.** Check `nvidia-smi` / `torch.cuda.is_available()` early. A
  CPU-only PyTorch install silently makes every run 10-25x slower without any error — if
  training time looks implausible for the model/data size, verify CUDA is actually wired up
  before assuming the hardware is just slow.
- **Kaggle datasets**: search for the canonical dataset slug, tell 김종민 the exact
  `kaggle.com/datasets/<owner>/<slug>` search term (and a size estimate if you can find one
  — but say so if you're not sure, size estimates from memory have been wrong before), then
  wait for him to report the local path. Verify the folder structure matches what you
  expected before building on it.
- **Long training runs**: launch with `run_in_background`, then stop polling and wait for
  the completion notification — don't `sleep`-poll. If you need to wait on something that
  genuinely isn't background-task-tracked, that's the one case for a scheduled check-in, not
  this.
- **Deliverable**: a self-contained HTML report (Tailwind CDN + Chart.js CDN, dark dashboard
  style consistent with earlier days in the same repo) is the default when the assignment
  wants a "report" or "리포트". A plain result-summary PNG is the default when the
  assignment explicitly asks for a "캡처"/screenshot rather than a report. When in doubt,
  check what a completed classmate submitted and match that format.

## Step 4 — Push to GitHub

1. `git pull --ff-only` in the local clone of the track's repo (e.g.
   `moyajong/oz-ai-06`) before adding anything.
2. Check the existing folder layout (`ls` the repo root) rather than assuming a naming
   convention — different tracks have used patterns like `computer-vision/day1/`,
   `data-analysis/day3/`, `deep-learning/day2/`. Match whatever pattern is already there for
   this track; kebab-case the track name if this is that track's first entry.
3. Copy the code + result files (report HTML, metrics JSON, key figures — not raw datasets)
   into that folder.
4. `git add` the specific files (not `-A`), commit, push. End the commit message with:
   ```
   Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_016G7vgQycimexbCUBGqQGUQ
   ```
   (pull the current session URL from context if it differs from this default).

## Step 5 — Embed the link in Notion and mark complete

1. Navigate back to 김종민's row for this day, open it.
2. Click the "파일 업로드 또는 임베드" placeholder under 필수 과제. A dialog opens with
   업로드/링크 tabs.
3. **Prefer the 링크 tab over 업로드.** Direct file upload requires interacting with a
   native OS file picker that browser automation can't reach — Notion creates the
   `<input type=file>` only transiently on click, so there's nothing stable to target. Paste
   the GitHub URL (the repo folder — `.../tree/main/<track>/dayN` — or, for an image
   deliverable, the raw file URL so it renders inline) into the 링크 tab and click
   "링크 임베드" instead of fighting the upload dialog.
4. Click the 상태 property, select 완료. Do this without asking for reconfirmation first —
   김종민 has already established that once a deliverable is genuinely built and pushed, 완료
   should follow automatically, not be gated on a second approval each time.
5. If any of these clicks don't register (Notion's UI sometimes silently no-ops a click),
   retry with a slightly different approach (exact coordinates from a fresh screenshot,
   `ref`-based click via `find`, or `scroll_to` then click) up to 2-3 times. If it's still
   stuck after that, stop and tell 김종민 exactly what's blocked rather than continuing to
   guess.

## Step 6 — Report completion

End with a short, concrete summary — this is 김종민's confirmation that the day is actually
done, not just attempted:

```
<N>일차 완료: 커밋 <short-hash> · <one-line result summary, e.g. "Test Accuracy 91.3%"> · Notion 완료 처리됨
```

## Data acquisition boundary

Never attempt to log into Kaggle, Google Drive, or any other account-gated service on
김종민's behalf, and never try to route around a login wall. When a required dataset needs
one, tell him the exact dataset name/search term and wait for him to download it and report
the local path — this holds regardless of how large or small the dataset is. This isn't
specific to this skill; it's the standing credential-handling rule for the whole session, but
it comes up on close to every OZ day, so it's worth restating here.

## Example

**Input:** "컴퓨터비전 2일차 진행하자"

**What happens:** Open the CV Day2 Notion page → find 김종민's row shows 시작 전 → open
"2일차 과제" → it's a free-form "실험한 이미지 분류 성능을 최대한 높게" prompt referencing an
external Colab practice notebook → build a real transfer-learning experiment reusing Day1's
dataset/split → GPU-train it in the background → build an HTML/PNG result summary → `ls` the
repo, find `computer-vision/day1/` already exists, add `computer-vision/day2/` → commit+push
→ open Day2's row again, embed the GitHub link via 링크 tab → set 상태 to 완료 → report
`"2일차 완료: 커밋 9c02b29 · Fine-tuning Test Accuracy 91.33% (Day1 대비 +7.83%p) · Notion 완료 처리됨"`.
