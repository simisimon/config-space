import re

MAPPING = {
    "vscode": [".vscode/launch.json", ".vscode/settings.json", ".vscode/tasks.json", ".vscode/extensions.json"],
    "github-action": [".github/workflows/*.yaml", ".github/workflows/*/*.yaml", ".github/workflows/*.yml", ".github/workflows/*/*.yml",  ".github/actions/*.yaml",  ".github/actions/*/*.yaml", ".github/actions/*.yml", ".github/actions/*/*.yml"],
    "apify": ["actor.json", "actor/*.json"],
    "buildkite": [".buildkite/pipeline.yml"],
    "eslint": [".eslintrc.json", ".eslintrc.yaml", ".eslintrc.yml", "*.eslint.json"],
    "codecov": ["codecov.yml", ".codecov.yaml", "codecov.yaml", ".codecov.yml"],
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
    "prettier": [".prettierrc.json", ".prettierrc.yaml", ".prettierrc.yml"],
    "amplify": ["amplify.yml"],
    "docker compose": ["*docker-compose.*.yml", "*docker_compose_*.yaml", "docker-compose.yaml"],
    "vercel": ["vercel.json"],
    "cirrus": [".cirrus.yml", ".cirrus.yaml"],
    "stripe": ["stripe-app.json"],
    "pnpm": ["pnpm-lock.yaml", "pnpm-workspace.yaml"],
    "turborepo": ["turbo.json"],
    "stylua": ["stylua.toml"],
    "cmake": ["CMakeSettings.json", "CMakePresets.json"],
    "drone": [".drone.yml", ".drone.yaml"],
    "github funding": [".github/FUNDING.yml", ".github/FUNDING.yaml"],
    "github issues": [".github/ISSUE_TEMPLATE/*.yml", ".github/ISSUE_TEMPLATE/*.yaml"],
    "github codespaces": [".github/codespaces/*.yml", ".github/codespaces/*.yaml"],
    "github config": [".github/config.yml", ".github/config.yaml"],
    "ansible": ["ansible.cfg", "ansible/*.cfg"],
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
    "pre-commit": [".pre-commit-config.yaml", ".pre-commit-config.yml", ".pre-commit-hooks.yaml", ".pre-commit-hooks.yml"],
    "snapscraft": ["snapcraft.yaml"],
    "spring": ["application-*.properties", "application-*.yml", "application-*.yaml"],
    "codebeaver": ["codebeaver.yml"],
    "vcpkg": ["vcpkg.json", "vcpkg-*.json"],
    "crowdin": ["crowdin.yml", "crowdin.yaml"],
    "readthedocs": ["readthedocs.yml", ".readthedocs.yml", ".readthedocs.yaml", "readthedocs.yaml"],
    "changesets": [".changeset/config.json"],
    "fuel": ["*/chainConfig.json", "*/metadata.json", "*/stateConfig.json"],
    "knip": ["knip.json"],
    "tsdoc": ["tsdoc.json"],
    "nodemon": ["nodemon.json", "nodemon.*.json"],
    "graphql": [".graphqlrc.yml", ".graphqlrc.yaml", "graphqlrc.json", "graphql.config.json", "graphql.config.toml", ".graphqlrc.toml"],
    "swagger": ["swagger.yaml", "swagger.yml", "swagger.json"],
    "grafana": ["grafana.ini", "grafana/*.ini"],
    "chart testing": ["chart_schema.yaml", "ct.yaml", "lintconf.yaml"],
    "nixpacks": ["nixpacks.json", "nixpacks.toml"],
    "codebuild": ["buildspec.yml", "buildspec.yaml", "buildspec_*.yml", "buildspec_*.yaml"],
    "lefthook": ["lefthook.yml", ".lefthook.yml", ".config/lefthook.yml", "lefthook.yaml", ".lefthook.yaml", ".config/lefthook.yaml", "lefthook.json", ".lefthook.json", ".config/lefthook.json", "lefthook.toml", ".lefthook.toml", ".config/lefthook.toml"],
    "bundlemon": [".bundlemonrc.json"],
    "hugoreleaser": ["hugoreleaser.yaml"],
    "deepsource": [".deepsource.toml"],
    "git cliff": ["git-cliff/cliff.toml"],
    "reuse": ["REUSE.toml"],
    "kodiak": [".kodiak.toml"],
    "streamlit": [".streamlit/config.toml", ".streamlit/secrets.toml"],
    "mdbook": ["mdbook.toml"],
    "lychee": ["lychee.toml"],
    "cdbindgen": ["cdbindgen.toml"],
    "triagebot": ["triagebot.toml", "triage/config.yml", "triage/config.yaml"],
    "taoskeeper": ["taoskeeper.toml"],
    "jazzy": [".jazzy.yaml"],
    "clomonitor": [".config/clomonitor/*.yaml", ".config/clomonitor/*.yml"],
    "prometheus": ["prometheus.yml", "prometheus.yaml"],
    "helm"  : ["Chart.yml", "Chart.yaml", "values.yaml", "values.yml", "charts/*.yaml", "charts/*.yml", "helm/*/values.yaml", "helm/*/values.yml", "helm/*/Chart.yaml", "helm/*/Chart.yaml"],
    "cspell": [".cspell.json", "cspell.json", "cspell.config.json", "cspell.config.yaml", "cspell.config.yml", "cspell.config.toml", "cspell.yaml", "cspell.yml"],
    "azure pipelines": ["azure-pipelines.yml", "azure-pipelines.yaml"],
    "logstash": ["logstash.yml"],
    "verdaccio": ["*/.verdaccio/config.yml"],
    "github": [".github/*.yml", ".github/*.yaml"],
    # CfgNet Mapping
    "alluxio": ["alluxio-site.properties"],
    "android": ["AndroidManifest.xml"],
    "angular": ["angular.json"],
    "ansible playbook": ["site.yml", "playbook.yml", "site.yaml", "playbook.yaml", "playbooks/*.yml", "playbooks/*.yaml"],
    "ansible": ["ansible.cfg"],
    "apache webserver": ["httpd.conf"],
    "cargo": ["Cargo.toml"],
    "circleci": [".circleci/config.yml"],
    "docker": ["Dockerfile"],
    "cypress": ["cypress.json"],
    "django": ["settings.py"],
    "docker compose": ["docker-compose.yml", "docker-compose.yaml"],
    "elasticsearch": ["elasticsearch.yml"],
    "flutter": ["pubspec.yaml"],
    "github action": [".*?\.github\/workflows\/[^\/]*\.yml$"], 
    "gradle": ["gradle.properties"],
    "gradle wrapper": ["gradle-wrapper.properties"],
    "hadoop common": ["core-site.xml"],
    "hadoop hbase": ["hbase-site.xml", "hbase-default.xml"],
    "hadoop hdfs": ["hdfs-site.xml", "hdfs-default.xml"],
    "heroku": ["Procfile"],
    "kafka": ["server.properties"],
    "kubernetes": ["log4j.properties", "log4j2.xml"],
    "maven": ["pom.xml"],
    "maven wrapper": ["maven-wrapper.properties"],
    "mapreducs": ["mapred-site.xml", "mapred-default.xml"],
    "mongodb": ["mongod.conf"],
    "mysql": ["my.cnf", "my.ini"],
    "netlify": ["netlify.toml"],
    "nginx": ["nginx.conf"],
    "nodejs": ["package.json"],
    "php": ["php.ini"],
    "postgresql": ["postgresql.conf"],
    "poetry": ["pyproject.toml"],
    "rabbitmq": ["rabbitmq.conf"],
    "redis": ["redis.conf"],
    "spring": ["application.properties", "application.yml", "application.yaml"],
    "travis": [".travis.yml"],
    "tsconfig": ["tsconfig.json"],    
    "yarn": ["yarn-site.xml"],
    "zookeeper": ["zoo.cfg"]
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
