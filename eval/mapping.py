import re

MAPPING = {
    "vscode": [".vscode/launch.json", ".vscode/settings.json", ".vscode/tasks.json", ".vscode/extensions.json"],
    "github-action": [".github/workflows/*.yaml", ".github/workflows/*/*.yaml", ".github/workflows/*.yml", ".github/workflows/*/*.yml",  ".github/actions/*.yaml",  ".github/actions/*/*.yaml", ".github/actions/*.yml", ".github/actions/*/*.yml"],
    "apify": ["actor.json", "actor/*.json"],
    "buildkite": [".buildkite/pipeline.yml"],
    "eslint": [".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml", "*.eslint.json"],
    "codecov": ["codecov.yml"],
    "dependabot": [".github/dependabot.yml", ".github/dependabot.yaml"],
    "markdownlint": [".markdownlint.json", ".markdownlint.yaml", ".markdownlint.yml"],
    "sourcery": [".sourcery.yaml"],
    "swiftlint": [".swiftlint.yml"],
    "yarn": [".yarnrc.yml"],
    "log4j": ["log4j2.properties", "log4j2.xml", "log4j.properties", "log4j.xml"],
    "firebase": ["firebase.json"],
    "lerna": ["lerna.json", "nx.json"],
    "cargo": [".cargo/config.toml"],
    "tsconfig": ["**/tsconfig.node.json", "**/tsconfig.decl.json", "**/tsconfig.base.json", "**/tsconfig.*.json"],
    "poetry": ["poetry.toml"],
    "mypy": ["mypy.ini"],
    "tox": ["tox.ini"],
    "rustfmt": ["rustfmt.toml", ".rustfmt.toml"],
    "mkdocs": ["mkdocs.yml", "mkdocs.yaml", "mkdocs.*.yml", "mkdocs.*.yaml"],
    "goreleaser": [".goreleaser.yml", ".goreleaser.yaml", ".goreleaser.*.yaml"],
    "codeclimate": [".codeclimate.yml", ".codeclimate.yaml"],
    "renovate": ["renovate.json", ".renovaterc.json"],
    "npm": ["package-lock.json"],
    "flyio": ["fly.toml"],
    "kubernetes": ["*/k8s/*.yaml", "k8s/*.yaml", "*/kubernetes/*/*.yaml", "*/kubernetes/*.yaml"],
    "taplo": ["taplo.toml", ".taplo.toml"],
    "bandit": ["bandit.yml", "bandit.yaml"],
    "prettier": [".prettierrc.json"],
    "amplify": ["amplify.yml"],
    "docker compose": ["*docker-compose.*.yml", "*docker_compose_*.yaml", "docker-compose.yaml"],
    "vercel": ["vercel.json"],
    "cirrus": [".cirrus.yml", ".cirrus.yaml"],
    "stripe": ["stripe-app.json"],
    "pnpm": ["pnpm-lock.yaml", "pnpm-workspace.yaml"],
    "turborepo": ["turbo.json"],
    "stylua": ["stylua.toml"],
    "cmake": ["CMakeSettings.json"],
    "drone": [".drone.yml", ".drone.yaml"],
    "github": [".github/ISSUE_TEMPLATE/*.yml", ".github/ISSUE_TEMPLATE/*.yaml", ".github/*.yml", ".github/*.yaml"],
    "rust": ["rust-toolchain.toml"],
    "cross": ["Cross.toml"],
    "cargo-make": ["Makefile.toml"],
    "trunk": ["Trunk.toml"],
    "golangci-lint": [".golangci.yml", ".golangci.yaml", ".golangci.toml", ".golangci.json"],
    "mockery": [".mockery.yaml"],
    "buf": ["buf.yaml", "buf.work.yaml", "buf.*.yaml"],
    "gitpod": [".gitpod.yml"],
    "appveyor": ["appveyor.yml", "appveyor.yaml"],
    "clippy": ["clippy.toml", ".clippy.toml"],
    "devcontainer": ["devcontainer.json"],
    "deno": ["deno.json", "deno.lock.json"],
    "bower": ["bower.json"],
    "babel": ["babel.config.json", ".babelrc.json"],
    "rubocop": [".rubocop.yml"],
    "gitbook": [".gitbook.yaml"],
    "stylelint": [".stylelintrc.json", ".stylelintrc.yml", ".stylelintrc.yml"],
    "postcss": [".postcssrc.json", ".postcssrc.yml"],
    "ruff": ["ruff.toml", ".ruff.toml"],
    "jitpack": ["jitpack.yml"],
    "gocrazy": ["gokrazy/*/config.json"],
    "jest": ["jest.config.json"],
    "jekyll": ["_config.yml", "_config.toml"],
    "mocha": [".mocharc.yaml", ".mocharc.yml", ".mocharc.json"],
    "typos": ["typos.toml", ".typos.toml", "_typos.toml"],
    "wrangler": ["wrangler.toml", "wrangler.json"],
    "pre-commit": [".pre-commit-config.yaml", ".pre-commit-config.yml"],
    "snapscraft": ["snapcraft.yaml"],
    "spring": ["application-*.properties", "application-*.yml", "application-*.yaml"],

}

def _match_pattern(pattern: str, filename: str) -> bool:
    """
    Glob-like matching:
      - patterns without '*' use simple suffix match
      - patterns with '*' are translated to regex:
          *  -> any chars except '/'
      - pattern may match as a substring anywhere in the path
    """
    # normalize to forward slashes in case of Windows-style paths
    filename = filename.replace("\\", "/")

    if "*" not in pattern:
        return pattern == filename or filename.endswith(pattern)

    # escape everything, then turn '\*' into '[^/]*'
    regex = re.escape(pattern).replace(r"\*", r"[^/]*")

    # use search, not fullmatch: allow any prefix directories
    return re.search(regex, filename) is not None


def get_technology(filename: str) -> str | None:
    is_matched = False
    for tech, patterns in MAPPING.items():
        for pattern in patterns:
            if _match_pattern(pattern, filename):
                is_matched = True
            
            if is_matched:
                return tech
    return None
