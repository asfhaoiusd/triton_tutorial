"""本机常缺 TeXGyreTermesX.otf / adforn；改成 DejaVu，并去掉装饰字体依赖。"""
from pathlib import Path
import re

p = Path(__file__).resolve().parent / "elegantbook.cls"
t = p.read_text(encoding="utf-8")
t = re.sub(
    r"\\setmainfont\{TeXGyreTermesX?\}\[[^\]]*\]",
    r"\\setmainfont{DejaVu Serif}",
    t,
    count=1,
    flags=re.S,
)
t = t.replace(r"\setmainfont{TeXGyreTermes}", r"\setmainfont{DejaVu Serif}")
t = re.sub(
    r"\\setsansfont\{texgyreheros\}\[[^\]]*\]",
    r"\\setsansfont{DejaVu Sans}[Scale=0.9]",
    t,
    count=1,
    flags=re.S,
)
t = t.replace(
    r"\setsansfont{TeXGyreHeros}[Scale=0.9]",
    r"\setsansfont{DejaVu Sans}[Scale=0.9]",
)
if r"\RequirePackage{adforn}" in t:
    t = t.replace(
        r"\RequirePackage{adforn}",
        r"\providecommand{\adftripleflourishleft}{}\providecommand{\adftripleflourishright}{}",
    )
p.write_text(t, encoding="utf-8")
print("patched fonts -> DejaVu", p)
