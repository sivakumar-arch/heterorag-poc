"""
Synthetic Stack-Exchange-style dump with an independently computed truth.

The truth is built from the generator's own data structures, not by re-reading the XML or by
using the oracle, so tests can check the oracle views and the gold functions against it.

The data deliberately contains the awkward cases of the real dump: ownerless questions, an
owner who is not a user, answers that precede their parent question, accepted answers,
links to answers and to missing posts, tags missing from the tag table, empty bodies, tokens
with underscores and apostrophes in titles, and heavy ties in reputation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import quoteattr

_SYL = ["ba", "ko", "ri", "ta", "mu", "ne", "so", "li", "pa", "ve", "du", "fi", "go", "ha", "ju", "ke"]


def _words(n: int, rng: random.Random) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add("".join(rng.choice(_SYL) for _ in range(rng.choice((3, 3, 4)))))
    return sorted(out)


@dataclass
class Truth:
    users: dict[int, dict] = field(default_factory=dict)             # loaded users (id > 0)
    questions: dict[int, dict] = field(default_factory=dict)          # id -> title, tags, owner, score, year
    answers: dict[int, dict] = field(default_factory=dict)            # id -> parent, owner
    asked: set[tuple[int, int]] = field(default_factory=set)          # (user, question), loaded users only
    answered: set[tuple[int, int]] = field(default_factory=set)
    answers_of: set[tuple[int, int]] = field(default_factory=set)     # (answer, question)
    accepted: set[tuple[int, int]] = field(default_factory=set)       # (question, answer)
    tagged: set[tuple[int, str]] = field(default_factory=set)         # (question, tag) with tag in tag table
    duplicates: set[tuple[int, int]] = field(default_factory=set)     # (question, original)
    linked: set[tuple[int, int]] = field(default_factory=set)
    badges: dict[int, int] = field(default_factory=dict)              # user -> badge count
    rare_badges: dict[str, set] = field(default_factory=dict)         # badge name -> holders (1 to 35)
    early_answers: int = 0


def _attr(**kw) -> str:
    return " ".join(f"{k}={quoteattr(str(v))}" for k, v in kw.items() if v is not None)


def make_synthetic_dump(d: Path, seed: int = 7, n_users: int = 300, n_questions: int = 1500) -> Truth:
    rng = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    t = Truth()
    vocab = _words(260, rng)
    weights = [1.0 / (i + 1) ** 0.9 for i in range(len(vocab))]
    tag_names = [f"tag-{w}" for w in _words(60, rng)]
    tag_w = [1.0 / (i + 1) ** 0.8 for i in range(len(tag_names))]
    in_table = set(tag_names[:55])                      # the last 5 are missing from the tag table

    # ---- users: heavy ties at 1 and 101, a thin tail
    rows = ['<row Id="-1" Reputation="1" DisplayName="Community" />']
    for u in range(1, n_users + 1):
        r = rng.choice([1] * 6 + [101] * 2) if rng.random() < 0.9 else int(rng.paretovariate(1.2) * 300)
        t.users[u] = {"reputation": r}
        about = ' AboutMe="hello there"' if rng.random() < 0.3 else ""
        rows.append(f'<row Id="{u}" Reputation="{r}" CreationDate="2012-0{rng.randint(1, 9)}-01T00:00:00.000" '
                    f'DisplayName="user{u}" Views="1" UpVotes="2" DownVotes="1"{about} />')
    (d / "Users.xml").write_text('<users>\n' + "\n".join(rows) + '\n</users>\n')

    # ---- posts: assign ids sequentially; some answers are emitted before their question
    seq: list[tuple] = []
    qids = []
    pid = 0
    pending_early: list[int] = []
    for _ in range(n_questions):
        pid += 1
        seq.append(("q", pid))
        qids.append(pid)
        if rng.random() < 0.02:
            pid += 1
            seq.append(("early", pid))
            pending_early.append(pid)
        for _a in range(rng.choice([0, 0, 1, 1, 2, 3, 7])):
            pid += 1
            seq.append(("a", pid))
    post_rows = []
    qset = set(qids)
    last_q = qids[-1]
    for kind, i in seq:
        owner = None
        r = rng.random()
        if r < 0.03:
            owner = None                               # ownerless
        elif r < 0.05:
            owner = n_users + 500                      # owner who is not a user
        else:
            owner = rng.randint(1, n_users)
        if kind == "q":
            ntag = rng.randint(1, 5)
            tags = list(dict.fromkeys(rng.choices(tag_names, tag_w, k=ntag)))
            title_words = rng.choices(vocab, weights, k=rng.randint(3, 8))
            title = " ".join(title_words)
            if rng.random() < 0.1:
                title += " x_1 don't"
            year = rng.choice([2015, 2016, 2017, 2018, 2019, 2020])
            score = int(rng.paretovariate(1.0)) - 1 + (rng.randint(0, 3) if rng.random() < 0.3 else 0)
            views = rng.randint(1, 5000)
            body = "" if rng.random() < 0.01 else f"<p>body {i}</p>"
            t.questions[i] = {"title": title, "tags": tags, "owner": owner, "score": score, "year": year,
                              "views": views, "body": body}
            tag_attr = "|" + "|".join(tags) + "|" if rng.random() < 0.8 else "".join(f"<{x}>" for x in tags)
            post_rows.append("<row " + _attr(
                Id=i, PostTypeId=1, CreationDate=f"{year}-03-04T05:06:07.000", Score=score, ViewCount=views,
                Body=body, OwnerUserId=owner, Title=title, Tags=tag_attr, AnswerCount=0, CommentCount=0) + " />")
            if owner in t.users:
                t.asked.add((owner, i))
            for x in tags:
                if x in in_table:
                    t.tagged.add((i, x))
        else:
            if kind == "early":
                parent = rng.choice([q for q in qids if q > i] or [last_q])
                if parent < i:                       # no later question exists: fall back to a normal answer
                    parent = rng.choice([q for q in qids if q < i])
                else:
                    t.early_answers += 1
            else:
                parent = rng.choice([q for q in qids if q < i])
            t.answers[i] = {"parent": parent, "owner": owner}
            t.answers_of.add((i, parent))
            if owner in t.users:
                t.answered.add((owner, i))
            post_rows.append("<row " + _attr(
                Id=i, PostTypeId=2, ParentId=parent, CreationDate="2020-05-06T07:08:09.000", Score=rng.randint(0, 9),
                Body=f"<p>answer {i}</p>", OwnerUserId=owner, CommentCount=0) + " />")
    # accepted answers (as extra attribute on question rows): patch rows
    by_parent: dict[int, list[int]] = {}
    for a, info in t.answers.items():
        by_parent.setdefault(info["parent"], []).append(a)
    patched = []
    for row in post_rows:
        if 'PostTypeId="1"' in row:
            qid = int(row.split('Id="')[1].split('"')[0])
            if qid in by_parent and rng.random() < 0.4:
                acc = rng.choice(by_parent[qid])
                t.accepted.add((qid, acc))
                row = row.replace("<row ", f'<row AcceptedAnswerId="{acc}" ', 1)
        patched.append(row)
    # a wiki post (type 5) used as tag wiki excerpt
    wiki_id = pid + 1
    patched.append(f'<row Id="{wiki_id}" PostTypeId="5" Body="wiki excerpt" />')
    (d / "Posts.xml").write_text('<posts>\n' + "\n".join(patched) + '\n</posts>\n')

    # ---- votes, badges, comments, links, tags
    (d / "Votes.xml").write_text('<votes>\n' + "\n".join(
        f'<row Id="{i}" PostId="{rng.choice(qids)}" VoteTypeId="2" CreationDate="2020-01-03T00:00:00.000" />'
        for i in range(1, 400)) + '\n</votes>\n')
    b_rows = []
    for i in range(1, 700):
        u = rng.choice(list(t.users))
        t.badges[u] = t.badges.get(u, 0) + 1
        b_rows.append(f'<row Id="{i}" UserId="{u}" Name="b{i % 9}" Date="2020-01-05T00:00:00.000" Class="3" TagBased="False" />')
    # rare badges (1 to 35 holders) from a separate generator, so the data drawn above is unchanged
    brng, bid = random.Random(seed + 101), 700
    for k, holders in enumerate((1, 1, 2, 2, 3, 4, 5, 8, 12, 15, 3, 2, 20, 25, 30, 35)):
        for u in brng.sample(sorted(t.users), holders):
            t.badges[u] = t.badges.get(u, 0) + 1
            t.rare_badges.setdefault(f"rare{k}", set()).add(u)
            b_rows.append(f'<row Id="{bid}" UserId="{u}" Name="rare{k}" Date="2020-01-05T00:00:00.000" Class="3" TagBased="False" />')
            bid += 1
    (d / "Badges.xml").write_text('<badges>\n' + "\n".join(b_rows) + '\n</badges>\n')
    (d / "Comments.xml").write_text('<comments>\n' + "\n".join(
        f'<row Id="{i}" PostId="{rng.choice(qids)}" Score="1" Text="comment {i}" CreationDate="2020-01-06T00:00:00.000" UserId="{rng.randint(1, n_users)}" />'
        for i in range(1, 300)) + '\n</comments>\n')
    l_rows, lid = [], 0
    for _ in range(300):
        a, b = rng.sample(qids, 2)
        kind = rng.choice([1, 1, 3])
        lid += 1
        l_rows.append(f'<row Id="{lid}" CreationDate="2020-05-01T00:00:00.000" PostId="{a}" RelatedPostId="{b}" LinkTypeId="{kind}" />')
        (t.duplicates if kind == 3 else t.linked).add((a, b))
    # links to an answer and to a missing post must not become graph edges
    some_answer = next(iter(t.answers))
    for rel in (some_answer, 987654):
        lid += 1
        l_rows.append(f'<row Id="{lid}" CreationDate="2020-05-01T00:00:00.000" PostId="{qids[0]}" RelatedPostId="{rel}" LinkTypeId="1" />')
    (d / "PostLinks.xml").write_text('<postlinks>\n' + "\n".join(l_rows) + '\n</postlinks>\n')
    tg = []
    for i, name in enumerate(tag_names[:55], 1):
        extra = f' ExcerptPostId="{wiki_id}" WikiPostId="{wiki_id}"' if i == 1 else ""
        tg.append(f'<row Id="{i}" TagName="{name}" Count="1"{extra} />')
    (d / "Tags.xml").write_text('<tags>\n' + "\n".join(tg) + '\n</tags>\n')
    return t
