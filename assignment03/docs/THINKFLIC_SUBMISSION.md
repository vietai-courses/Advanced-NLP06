# ThinkFlic Final Submission

This document describes the final assessment package uploaded to ThinkFlic after the Kaggle competition. It is separate from the Kaggle `submission.csv` workflow.

## Deadline and Submitter

- **Deadline:** July 11, 2026 at 23:59 (Asia/Saigon), two days after the Kaggle deadline.
- **Submit on ThinkFlic:** <https://course.newturing.ai/courses/take/nlp-agent/assignments/76173989-assignment-03-self-improving-ai-for-financial-question-answering>
- Each team submits exactly one ZIP package.
- The team chooses one member as the designated ThinkFlic submitter.
- Individual participants submit their own package.

Name the ZIP:

```text
A3_<KaggleTeamName>_<StudentEmail1>_<StudentEmail2>.zip
```

For an individual submission, omit `<StudentEmail>`.

## Local Files vs. ThinkFlic Folder

These are two different locations:

1. **Your local assignment workspace** is the folder containing `run_modal.py`, `graders/`, and `src/`. Run `modal run run_modal.py::get_proof` from this folder. It downloads the Modal output into `runs/exp_self/` and also copies `evolution_proof.json` beside `run_modal.py` so `python graders/grade_stage4_harness.py` can validate it.
2. **Your ThinkFlic submission folder** is a new folder that you create only for the final ZIP upload. The structure below describes this submission folder; it is not the repository root and should not replace your working assignment directory.

## Required Folder Structure

```text
A3_<KaggleTeamName>_<StudentEmail1>_<StudentEmail2>/
|-- README.md
|-- report.pdf
|-- integrity_declaration.pdf
|
|-- source_code/
|   |-- src/
|   |-- requirements.txt
|   `-- run_instructions.md
|
|-- kaggle/
|   |-- final_submission.csv
|   `-- submission_information.txt
|
`-- evidence/
    |-- evolution_proof.json       <- copy from ./evolution_proof.json
    |-- failure_mode_report.pdf    <- copy from ./runs/exp_self/failure_mode_report.pdf
    |-- learning_curve.pdf         <- copy from ./runs/exp_self/learning_curve.pdf
    `-- strategy_diversity.pdf     <- copy from ./runs/exp_self/strategy_diversity.pdf
```

Paths after `<-` show where each generated file is normally found relative to the local assignment workspace. They are explanatory annotations, not part of the filenames.

Do not add comments inside `evolution_proof.json`: JSON does not support comments, and the grader expects the generated structure. Copy the file unchanged.

Do not include a notebook in `source_code/`. Submit the implemented `src/` directory, dependency list, and exact reproduction instructions.

## Required `README.md` Information

- Kaggle team name.
- Full name, student ID, class, and Kaggle username for every member.
- Designated ThinkFlic submitter and each member's contribution.
- Final Kaggle submission name and available leaderboard score/rank.
- Best development accuracy and repository commit hash.
- Google Drive video link with access set to **Anyone with the link can view**.
- All provided, external, and synthetic data used.

## Technical Report

Submit `report.pdf`, recommended length **4-6 pages** excluding appendices:

1. Team information and member contributions.
2. Executive summary and final Kaggle result.
3. Final model, prompt, training, ensemble, and post-processing pipeline.
4. External and synthetic data sources, generation procedure, and approximate volume.
5. Validation experiments, ablations, failure analysis, and final-system selection.
6. Reproduction commands, dependencies, model/API settings, compute usage, approximate cost, and commit hash.
7. Limitations and lessons learned.

## Video Presentation

Provide a **5-8 minute** Google Drive video link in both `README.md` and `report.pdf`. Introduce the team, demonstrate the final pipeline, summarize experiments and data usage, show the Kaggle result, and describe each member's contribution. Do not include the video file in the ZIP.

## Integrity Declaration

Both members must sign or type their names beneath this statement in `integrity_declaration.pdf`:

> We confirm that our team did not retrieve, reconstruct, manually label, or share answers for specific competition test samples. We did not use matched transcripts, subtitles, source documents, metadata, another team's predictions, or leaderboard probing to obtain test labels. All external and synthetic data used by our system is disclosed in the report.

## Final Checklist

- [ ] ZIP filename follows the required convention.
- [ ] All students and contributions are identified.
- [ ] `report.pdf` and `integrity_declaration.pdf` are included.
- [ ] `source_code/src/`, dependencies, and reproduction instructions are included.
- [ ] The final Kaggle submission and submission information are included.
- [ ] All four required evidence files are included.
- [ ] The Google Drive video link works without requesting access.
- [ ] External and synthetic data, compute, API usage, and approximate cost are disclosed.

An incomplete, inaccessible, or unverifiable package may be returned for correction or ruled ineligible for Phase 3 points.
