# MaiBot Mashiro

This fork maintains a small set of conversation-quality fixes on top of MaiBot. The initial `mashiro` branch starts at upstream tag `1.3.3` (`cb692c92`). The upstream license and attribution are retained.

## Branches and updates

- `mashiro` is the maintained customization branch and the fork's default branch.
- `main` is retained as an upstream reference, not the deployment branch.
- `upstream` refers to `https://github.com/Mai-with-u/MaiBot.git`.
- Upstream tags keep their original meanings. They are not customized releases of this fork.

The server's current local settings are separate from the source repository. API keys, account IDs, `.env`, chat logs, databases and media are not included. `profiles/mashiro.toml` is a partial settings example; it must not replace an entire existing configuration file.

## Maintained changes

1. Quote previews explicitly preserve the original author and the boundary between quoted text and the new message, including incoming cached quotes and outgoing messages.
2. The normal Replyer path uses actual chat records, preserves self-message metadata, and receives original images up to the configured limit. Full Planner analysis is kept for diagnostics instead of becoming the Replyer's script. Use `replyer_retro_prompt = false`.
3. Planner tool instructions and Replyer prose instructions are separated. Necessary facts can still be provided through `reply_reference`.
4. Full timestamps and the actual triggering message, with the gap from the preceding chat message, identify the current turn. Old topics remain available as context and can be explicitly resumed. This is contextual guidance, not a fixed age-based ban on replies.
5. Responses function definitions explicitly set `strict = false` to preserve optional attachment arguments.

The optional profile pauses automatic expression selection without deleting learned expressions and sets the emotion trait to neutral. Model selection, API keys, voice configuration, vector retrieval and reply frequency remain deployment choices.

## Upgrade procedure

Work in a clean integration checkout. Keep the running deployment at its tested commit until the integration is verified.

```sh
git fetch upstream --tags
git switch -c integration/next-release mashiro
# Replace <release-tag> with the upstream version being reviewed.
git merge --no-ff <release-tag>
```

Resolve conflicts by preserving the behavior described above in the new architecture. Do not copy old implementation files over newer upstream files. If upstream already implements a fix, remove the redundant local change and retain its regression test.

Run the focused tests:

```sh
uv sync --frozen --group dev
uv run --frozen pytest -q pytests/mashiro pytests/maisaka/test_planner_messages.py pytests/maisaka/test_context_history.py
```

Also review changed schemas, configuration migrations, plugin compatibility and model request shapes. Textually clean merges can still change behavior. Check a few offline dialogue cases before promoting the integration branch into `mashiro` and deploying that reviewed commit.

Do not use an official one-click updater or force-sync the customized branch to upstream: those paths do not maintain this fork's commits. Configuration usually survives normal upgrades, but renamed or removed settings still need review. Keep the tested code version and deployment data backup together when an upgrade migrates databases.

## Verification boundary

The regression suite uses synthetic messages and settings, without QQ delivery or model credentials. It checks quote attribution, message metadata, Planner/Replyer separation, image limits, current-turn context, translated placeholders and optional tool arguments. It cannot guarantee that an LLM will understand every joke or consistently make the desired conversational choice.

The GitHub Actions workflow runs these checks for changes to `mashiro`. When creating a GitHub fork, Actions may need to be enabled for that fork before checks can run.
