---
layout:     post
title:      用Claude Code从零搭建自动化URL发现与语义过滤系统
subtitle:
date:       2026-08-11
author:     Schoko
header-img:
catalog: true
tags:
    - Claude Code
    - Python
    - 爬虫
    - NLP
---

### 前言

前段时间，我投了一家叫 **WLDM** 的公司，参与他们 full remote Data Scientist 职位的招聘流程。WLDM 是一家远程办公的数字营销公司，base在海外，团队分布在各个国家。他们的业务覆盖 SEO、CRO、PPC、Backlinks 等领域，客户遍布多个行业。

招聘流程中有一个 trial task 环节：给一份含约 3000 个噪音根域名和 961 个目标关键词的数据集（niche 为赌博/casino 领域），要求从这些域名里找到具体的深层页面URL，并生成带有相关度打分的筛选结果。换句话说，输入两张表：`input_domains.csv` + `target_keywords.csv`，输出一张干净的匹配结果表。

这个 trial task 全程用 Claude Code 开发完成。整个开发过程，角色分工大概是这样的：
- **人**：定义每个阶段的目标、review 产出、给出反馈
- **Claude Code**：写代码、执行、分析结果、写文档

也是在这期间，我第一次体验了 full remote 的工作方式——和来自世界各地的工程师开视频会议、参加 weekly meeting、和 manager 交流业务细节。WLDM 的团队风格很务实，对数据驱动业务的理念贯彻得比较彻底。虽然最后没有加入他们，但这段经历让我对 remote work 有了真实的感受，挺有意思的。

这篇博客记录一下技术实现过程，也给想用 Claude Code 做项目的同学一个参考。

### Pipeline总体架构

整个pipeline的设计思路算是比较直接的：从域名出发 → 发现站内URL → 抓取页面内容 → 关键词匹配 → 打分过滤 → 导出。但每一步都有一些工程上需要处理的细节。

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  加载数据    │───▶│  URL 发现     │───▶│  内容提取     │───▶│  关键词匹配  │
│  + 去重     │    │  (Sitemap +  │    │  (Title/H1/  │    │  (三阶段)    │
│             │    │   首页)       │    │   Body)      │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  导出 CSV   │◀───│  打分与过滤  │◀───│  工具页过滤  │◀───│  原始匹配    │
│             │    │  (0-100分)   │    │              │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

技术选型：Python 3.12 + aiohttp 异步爬虫 + BeautifulSoup 解析 + scikit-learn TF-IDF 做文本匹配。没有用Scrapy之类的大框架，因为这里的爬取逻辑并不复杂——大部分时间花在sitemap解析和HTML内容提取上，用aiohttp搭配asyncio信号量做并发控制就够了。

### 第一步：URL发现——高召回率优先

第一阶段的重点：宁可多抓，不能漏抓。

#### 爬虫设计

入口是 `run_pipeline.py`。对每个域名，先尝试解析sitemap（robots.txt + 常见sitemap路径），sitemap拿不到内容时回退到解析首页的链接：

{% highlight python %}
from crawler_utils import crawl_domain, fetch_page, extract_page_content, should_skip_url

async def process_single_domain(
    session: aiohttp.ClientSession,
    domain: str,
    matcher: KeywordMatcher,
    semaphore: asyncio.Semaphore,
) -> Tuple[List[dict], Optional[dict]]:
    results = []
    async with semaphore:
        try:
            discovered_urls = await crawl_domain(
                session, domain, max_urls=MAX_URLS_PER_DOMAIN
            )
            if not discovered_urls:
                return [], {"domain": domain, "error": "No URLs discovered"}

            urls_to_fetch = discovered_urls[:MAX_PAGES_TO_CRAWL]
            tasks = [fetch_page(session, url) for url in urls_to_fetch]
            html_results = await asyncio.gather(*tasks, return_exceptions=True)
            # ...
{% endhighlight %}

几个关键的工程参数：

- 每个域名最多发现500个URL，最多抓取其中200个页面
- 50个并发连接，每个域名间隔1.5秒（避免被封）
- 关闭SSL验证——真实世界的域名，证书过期的、自签的一大把
- 通过本地HTTP代理访问（`http://127.0.0.1:8890`）

#### 工具页过滤

抓取页面后，先做一轮工具页过滤。这一步虽简单但效果很好——URL泄露率0%，标题泄露率0.37%：

{% highlight python %}
def is_utility_page(url: str, title: str, h1_text: str) -> bool:
    """Check if a page is a utility/non-content page."""
    utility_patterns = [
        "about", "contact", "privacy", "terms", "login", "register",
        "signup", "signin", "cookie", "disclaimer", "faq", "help",
        "search", "sitemap", "careers", "jobs", "investor",
        "forum", "community", "press", "media", "news",
        "author", "contributor", "tag/", "category/",
    ]
    # ...
{% endhighlight %}

`should_skip_url`在URL发现阶段就拦截了一波，`is_utility_page`在内容提取后做第二层过滤。两层防护下来，About Us、Privacy Policy这类页面基本进不了匹配环节。

#### 第一阶段结果

处理了约1850个域名，1188个域名有匹配结果，产生了34461条原始URL-关键词匹配对。但也暴露了问题：**高召回率意味着高噪声**。

比如关键词列表里有 `action`、`space`、`national`、`party` 这种通用英文词。它们确实也是赌博领域会用到的词（sports action、party casino等），但在非赌博网站上也能匹配到一大堆——ibtimes.co.uk、benzinga.com这种新闻站就靠 `action` 这个词命中了上千条。实际上95.8%的记录得分都是1.0（精确匹配），分数完全没有区分度。

### 第二步：精确率提升——从34461到15102

第一阶段的产出质量不够，所以第二阶段的目标是**提升精确率**。这一步我们分了四个子任务来处理。

#### 2.1 关键词画像分析

先搞清楚我们的关键词到底长什么样。Claude Code帮我写了个分析脚本 `classify_keywords.py`，把所有关键词按模式匹配分成了11个类别：

| 类别 | 数量 | 占比 |
|------|------|------|
| 老虎机游戏名称 | 298 | 31.0% |
| 其他/未分类（多为老虎机名称） | 410 | 42.7% |
| 体育联赛/赛事 | 75 | 7.8% |
| 赌场/桌面游戏 | 41 | 4.3% |
| 赌场指南/玩法教程 | 34 | 3.5% |
| 通用/模糊词 | 17 | 1.8% |

关键发现：73.7%的关键词是老虎机相关的，真正的噪声词只有17个（1.8%）。这说明问题不在于关键词不好，而在于匹配策略太简单。

#### 2.2 三级级联匹配器

第一阶段的匹配只有简单的子串匹配，这显然不够。于是改成了三级级联匹配——`matcher_utils.py` 中的 `KeywordMatcher` 类：

{% highlight python %}
class KeywordMatcher:
    def __init__(self, keywords, tfidf_threshold=0.18):
        self.keywords = keywords
        self.tfidf_threshold = tfidf_threshold
        self.stemmer = SnowballStemmer("english")
        # 多语言词干提取：英语 + 法语/西语/葡萄牙语
        self.stemmer_map = {
            "en": self.stemmer,
            "fr": FrenchStemmer(),
            "es": SpanishStemmer(),
            "pt": PortugueseStemmer(),
        }
        self._build_tfidf_index()
{% endhighlight %}

三个匹配阶段是层进的：

1. **精确子串匹配**（Phase 1）—— 关键词直接出现在文本中，速度最快
2. **词元Jaccard匹配**（Phase 2）—— 对关键词和目标文本分别做词干提取和小写化后计算Jaccard系数，能匹配到变形形式（复数、时态等）
3. **TF-IDF余弦相似度**（Phase 3）—— 用scikit-learn的TfidfVectorizer，对所有关键词构建TF-IDF索引，对页面文本做同样的向量化后计算余弦相似度

{% highlight python %}
def match(self, text, url, title, h1_text):
    matches = []

    # Phase 1: exact substring
    for kw in self.keywords:
        if kw.lower() in text_lower:
            matches.append((kw, 1.0))

    # Phase 2: token-level Jaccard
    doc_tokens = set(self._tokenize(text))
    for kw in self.keywords:
        kw_tokens = set(self._tokenize(kw))
        jaccard = len(kw_tokens & doc_tokens) / len(kw_tokens | doc_tokens)
        if jaccard >= self.jaccard_threshold:
            matches.append((kw, jaccard))

    # Phase 3: TF-IDF cosine similarity
    if doc_tokens:
        doc_text = " ".join(doc_tokens)
        doc_vec = self.vectorizer.transform([doc_text])
        cosine_sim = cosine_similarity(doc_vec, self.kw_matrix)[0]
        for i, score in enumerate(cosine_sim):
            if score >= self.tfidf_threshold:
                matches.append((self.keywords[i], score))
    # ...
{% endhighlight %}

TF-IDF阈值设在0.18是经过多次调整的。太低会引入噪声，太高会漏掉语义相关但用词不同的页面。

#### 2.3 四层主题相关性打分

有了三级匹配还不够。和Claude Code讨论后，设计了四层0-100分的相关性打分体系——`filter_results.py`：

{% highlight python %}
def compute_relevance_score(row, keyword_specificity, domain_blacklist):
    score = 0.0

    # L1: 关键词特异性 (0-40分)
    kw = row["Detected Keyword"].lower()
    if any(kw in slot_kw for slot_kw in SLOT_KEYWORDS):
        # 老虎机关键词，特异性最高
        score += 40
    elif any(kw in guide_kw for guide_kw in GUIDE_KEYWORDS):
        score += 35
    elif any(kw in casino_kw for casino_kw in CASINO_KEYWORDS):
        score += 30
    else:
        # 通用词，惩罚较大
        score += max(10, keyword_specificity.get(kw, 10))

    # L2: 域名相关性 (0-30分)
    domain = row["Source Domain"].lower()
    if domain in domain_blacklist:
        # 黑名单域名，强制上限29分
        return min(score, 29)
    # 检查域名中的赌博信号词
    gambling_signals = [
        "casino", "bet", "slot", "poker", "gambling",
        # ... 更多信号词
    ]
    score += sum(5 for s in gambling_signals if s in domain)

    # L3: 内容质量 (0-20分)
    url_depth = row["Target URL"].count("/") - 2
    # URL深度加分 + 标题丰富度 + 路径赌博信号检测
    score += min(url_depth * 2, 8)

    # L4: 匹配质量 (0-10分)
    # 多词关键词得分更高
    if len(kw.split()) > 2:
        score += 5
    elif len(kw.split()) > 1:
        score += 3
    else:
        score += 1

    return min(score, 100)
{% endhighlight %}

分档逻辑：

| 分档 | 分值范围 | 记录数 | 占比 |
|------|----------|--------|------|
| HIGH | ≥ 70 | 2,728 | 7.9% |
| MEDIUM | 50–69 | 12,374 | 35.9% |
| LOW | 30–49 | 8,035 | 23.3% |
| NOISE | < 30 | 11,324 | 32.9% |

**34,461 → 15,102条**（保留率43.8%），黑名单域名全部被剔除（52个黑名单域名在最终交付物中0条），通用噪声词全部归入LOW/NOISE。

#### 2.4 错误分析与重试

662个域名处理失败，其中77.0%是"未发现任何URL"（基本上是非赌博域名，sitemap拿不到或首页挂了），23.0%是"有URL但无关键词匹配"。

Claude Code帮我分析后发现，有64个域名实际上是赌博相关域名但失败了（sitemap/首页不可达），于是写了 `retry_errors.py` 专门重试这些：

`retry_errors.py` 的改进策略：多个 UA 轮换，外加额外路径尝试和 HTTP/HTTPS 双协议回退。

{% highlight python %}
USER_AGENTS = [
    "Mozilla/5.0 ...",
    "Mozilla/5.0 ...",
]
{% endhighlight %}

最终回收到1个域名（online-kolikkopelit.fi），追加了27条结果。

### 第三步：代码整合与交付

按照需求方要求，最终交付一个自包含的 `.ipynb` notebook。`generate_notebook.py` 把分散在多个 `.py` 模块中的代码整合成一个 notebook，包含8个Section，所有Markdown单元格中英双语。

最终交付物：
- `url_discovery_final.ipynb` — 核心代码交付
- `results_final.csv` — 15,102条高质量匹配结果
- `project_documentation_en.md` / `cn.md` — 中英文项目文档

### Claude Code协作体验

整个项目做了大概几天（业余时间），说实话**比我自己从头写快太多了**。说说几个真实感受：

**好的方面**：

1. **代码生成质量很高**。我描述清楚需求，它写出来的代码基本可以直接跑，不需要大改。`matcher_utils.py`的三阶段匹配器，一次生成就很接近最终版本。

2. **适合多轮迭代**。对于探索性的任务——比如"先跑一轮看看结果"、"分析下结果有什么问题"、"设计一个打分策略把噪声过滤掉"——Claude Code特别适合。它能看到上一轮的结果，然后基于数据做调整。反馈循环很短。

3. **文档能力出人意料**。最后的项目文档（中英文），Claude Code读了一遍所有代码和中间结果，生成的项目文档比我自己写要条例清晰得多。

**需要注意的方面**：

1. **人工review不可省略**。每一步产出我都要看一遍。不是说代码会有语法错误，而是业务逻辑上可能有隐患。比如打分体系里L3"内容质量"那一层，URL深度的权重设多少合理，这个只有人根据业务来定。

2. **给Claude Code的指令需要比较明确**。模糊的指令会得到模糊的结果。越具体越好——"分析errors.csv中域名失败的原因，判断哪些值得重试"，比"看看错误"有效得多。

3. **Context很重要**。每一步在发起新任务时，我尽量同步上下文，比如当前的输出文件列表、之前讨论的要点。这样它不会跑偏。

### 总结

这个 trial task 的核心问题不复杂：**从一堆噪音域名里找出跟关键词有关的具体页面**。但工程上需要处理的细节不少——异步抓取、证书问题、多语言词干提取、语义匹配、打分过滤、错误重试。

用 Claude Code 开发这类数据处理 pipeline 项目，效率提升明显。人的精力集中在"这个阶段要达成什么目标"和"这个结果对不对"，细节实现和重复性工作交给 AI。

抛开技术部分，这次经历让我第一次真正体验了 full remote work——跨国团队协作、英文视频会议、不同时区的人一起过 weekly sync。WLDM 的团队氛围和业务理念让我印象不错。虽然最终没加入，但整个过程挺有收获的。如果你在用 Claude Code 做类似的事情，或者在 remote 环境里做 DS 工作，欢迎交流。

完整项目地址：[project-automated-URL-discovery-and-semantic-relevance-filter](https://github.com/sesebuckin/project-automated-URL-discovery-and-semantic-relevance-filter)
