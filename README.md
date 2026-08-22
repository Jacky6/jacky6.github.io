# jacky6.github.io

基于 [MkDocs + Material 主题](https://squidfunk.github.io/mkdocs-material/) 的个人博客站点，
线上地址：**https://jacky6.github.io/**

学习文档的源内容在独立仓库 [blogs](https://github.com/Jacky6/blogs)，
以 **git submodule** 的方式挂载在本仓库的 `docs/blogs` 目录下。

## 目录结构

```
.
├── mkdocs.yml          # 站点配置（主题 + 导航结构）
├── docs/
│   ├── index.md        # 首页
│   └── blogs/          # submodule → Jacky6/blogs（文档源）
├── site/               # 构建产物（已忽略，勿提交）
├── pyproject.toml      # 依赖：mkdocs >= 1.6.1, mkdocs-material >= 9.7.6
└── .python-version     # Python 3.12（用 uv 管理）
```

## 分支说明

| 分支 | 用途 |
|------|------|
| `dev/mkdoc_version` | **工作分支**，存放 `mkdocs.yml`、submodule 指针等源文件 |
| `gh-pages` | GitHub Pages 读取的部署分支，由 `mkdocs gh-deploy` 自动生成，**不要手动改** |
| `master` | 历史遗留，不再维护 |

## 环境准备（首次）

```shell
# install dependencies with uv (Python 3.12)
uv sync
```

## 日常操作流程

### 1. 更新文档源（blogs 仓库有新内容时）

```shell
cd docs/blogs
git pull origin main        # pull latest docs
cd ../..
```

### 2. 本地预览

```shell
uv run mkdocs serve         # live preview at http://127.0.0.1:8000
```

### 3. 部署到 GitHub Pages

```shell
uv run mkdocs gh-deploy --force
# build site/ → commit to gh-pages → push, Pages auto-deploys in ~1 min
```

### 4. 提交本仓库的源文件变更

部署只更新了 `gh-pages` 分支，**工作分支上的改动还要单独提交**，
否则换机器后 submodule 指针会丢失：

```shell
git add mkdocs.yml docs/blogs       # docs/blogs is the submodule pointer
git commit -m "update docs to blogs@<commit>"
git push origin dev/mkdoc_version
```

> 完整顺序建议：改文档 → push blogs → 本仓库 `git pull` submodule →
> `gh-deploy` → 提交并推送本仓库。

## 修改导航（新增板块）

编辑 `mkdocs.yml` 的 `nav`，路径相对于 `docs/`，指向对应目录的 `README.md`：

```yaml
nav:
  - 首页: index.md
  - 算法: blogs/programming/algorithm/README.md
  - langchain: blogs/llm/langchain/README.md
  - deep-agents: blogs/llm/langchain/deep-agents/README.md
  - langgraph: blogs/llm/langchain/langgraph/README.md
  - python: blogs/programming/python/cookbook/README.md
  - java: blogs/programming/java/README.md
```

> ⚠️ blogs 仓库的目录结构会变（如 `python_claw` → `python/cookbook`），
> 重构目录后记得同步更新这里的 `nav` 路径，否则构建出的导航是死链。

## 已知问题

- `mkdocs build --strict` 会失败：`KNOWLEDGE.md`、`java/java.md` 等
  Notion 导出残留文件里有大量失效链接（约 16 处）。日常构建不加
  `--strict` 即可，这些页面也未列入导航。
- blogs 里的 `llm/build-your-own-coder/` 是未提交的内嵌独立仓库，
  不会进入本仓库的 submodule 检出，因此**不会**发布到站点；
  如果以后想发布，需要先在 blogs 里提交它。
