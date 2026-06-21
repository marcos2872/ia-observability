"""Gera apresentacao PPTX futurista — Observabilidade em IA.

Paleta neon-cyber / dark-mode com 23 slides, capa com imagem full-bleed.
"""

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# ---------------------------------------------------------------------------
# Paleta cyber-futurista
# ---------------------------------------------------------------------------
BG_PRIMARY = RGBColor(0x07, 0x0B, 0x14)  # fundo escuro espacial
BG_CARD = RGBColor(0x12, 0x17, 0x28)  # card bg
BG_CARD2 = RGBColor(0x18, 0x1E, 0x30)  # card alt
CYAN = RGBColor(0x00, 0xD4, 0xFF)  # neon primario
PURPLE = RGBColor(0x7C, 0x3A, 0xED)  # neon secundario
PINK = RGBColor(0xFF, 0x00, 0x6E)  # terciario
MINT = RGBColor(0x00, 0xF5, 0xA0)  # success
TEXT_LIGHT = RGBColor(0xE8, 0xEA, 0xED)  # texto principal
TEXT_MUTED = RGBColor(0x8B, 0x8F, 0x9E)  # texto secundario
CODE_BG = RGBColor(0x0D, 0x11, 0x1C)  # fundo de codigo
BORDER = RGBColor(0x1E, 0x24, 0x3A)  # borda sutil
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY_DOT = RGBColor(0x3A, 0x40, 0x55)

CARD_COLORS = [CYAN, PURPLE, MINT, PINK]  # ciclo para cards

# ---------------------------------------------------------------------------
# Constantes de layout
# ---------------------------------------------------------------------------
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
M = Inches(0.8)  # margem lateral
CONTENT_W = Inches(11.7)  # largura util
TITLE_SIZE = 32
SUBTITLE_SIZE = 16
BODY_SIZE = 16
CODE_SIZE = 13
SMALL_SIZE = 10


# ===================================================================
# Helpers de baixo nivel
# ===================================================================


def _set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _alpha(shape, val):
    """Ajusta opacidade (0=invisivel, 100000=opaco) via XML."""
    solidFill = shape.fill._fill._solidFill  # CT_SolidColorFillProperties
    srgb = solidFill.find(qn("a:srgbClr"))
    if srgb is None:
        for child in solidFill:
            if child.tag == qn("a:srgbClr"):
                srgb = child
                break
    if srgb is None:
        return
    for a in srgb.findall(qn("a:alpha")):
        srgb.remove(a)
    alpha = etree.SubElement(srgb, qn("a:alpha"))
    alpha.set("val", str(val))


def _rect(slide, l, t, w, h, color, alpha_pct=None):
    """Retangulo solido, com alpha opcional (0-100)."""
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    if alpha_pct is not None:
        _alpha(s, int(100000 * alpha_pct / 100))
    return s


def _txt(
    slide,
    l,
    t,
    w,
    h,
    text,
    size=BODY_SIZE,
    bold=False,
    color=TEXT_LIGHT,
    align=PP_ALIGN.LEFT,
    font="Calibri",
):
    """Caixa de texto simples."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font
    p.alignment = align
    return tf


def _rich_box(slide, l, t, w, h):
    """TextFrame para construcao rica."""
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def _bullet(
    tf, text, size=BODY_SIZE, color=TEXT_LIGHT, bold=False, indent=0, sb=Pt(6), sa=Pt(2)
):
    """Adiciona paragrafo com bullet."""
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.font.bold = bold
    p.level = indent
    p.space_before = sb
    p.space_after = sa
    return p


def _run(p, text, size=BODY_SIZE, bold=False, color=TEXT_LIGHT, font="Calibri"):
    """Adiciona run a um paragrafo existente."""
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = font
    return r


def _code(slide, l, t, w, h, lines, size=CODE_SIZE):
    """Bloco de codigo com fundo escuro."""
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = CODE_BG
    s.line.fill.background()
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(16)
    tf.margin_right = Pt(16)
    tf.margin_top = Pt(14)
    tf.margin_bottom = Pt(14)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size = Pt(size)
        p.font.name = "Consolas"
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(2)
        p.space_after = Pt(2)
    return s


def _page(slide, num, total=23):
    _txt(
        slide,
        Inches(12.2),
        Inches(7.05),
        Inches(1),
        Inches(0.35),
        f"{num:02d}/{total}",
        size=8,
        color=TEXT_MUTED,
        align=PP_ALIGN.RIGHT,
    )


def _accent_bar(slide, y, color=CYAN):
    """Linha horizontal fina no topo."""
    _rect(slide, Inches(0), y, SLIDE_W, Pt(3), color)


def _section_number(slide, num, label, subtitle=""):
    """Slide de secao com numero grande e label."""
    _set_bg(slide, BG_PRIMARY)
    _rect(slide, Inches(0), Inches(0), SLIDE_W, Pt(4), CYAN)
    # numero grande
    _txt(
        slide,
        M,
        Inches(2.0),
        Inches(4),
        Inches(1.5),
        f"0{num}" if num < 10 else str(num),
        size=72,
        bold=True,
        color=CYAN,
        align=PP_ALIGN.LEFT,
    )
    # label
    _txt(
        slide,
        M,
        Inches(3.6),
        Inches(10),
        Inches(0.8),
        label,
        size=36,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.LEFT,
    )
    if subtitle:
        _txt(
            slide,
            M,
            Inches(4.3),
            Inches(10),
            Inches(0.5),
            subtitle,
            size=15,
            color=TEXT_MUTED,
            align=PP_ALIGN.LEFT,
        )
    # linha decorativa diagonal (simulada com retangulo fino)
    _rect(slide, M, Inches(4.8), Inches(6), Pt(2), RGBColor(0x2A, 0x30, 0x48))


def _card(slide, l, t, w, h, accent_color=CYAN):
    """Card com borda de accent na esquerda."""
    c = _rect(slide, l, t, w, h, BG_CARD)
    c.line.color.rgb = BORDER
    c.line.width = Pt(1)
    # barra lateral
    _rect(slide, l, t, Pt(4), h, accent_color)
    return c


# ===================================================================
# Construtor principal
# ===================================================================


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    BLANK = prs.slide_layouts[6]

    def slide():
        return prs.slides.add_slide(BLANK)

    # ================================================================
    # SLIDE 1 — CAPA (full-bleed imagem + overlay)
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    # full-bleed imagem
    img_path = "docs/images/Gemini_Generated_Image_6oaua56oaua56oau.png"
    s.shapes.add_picture(img_path, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    # overlay escuro 55%
    _rect(s, Inches(0), Inches(0), SLIDE_W, SLIDE_H, BG_PRIMARY, alpha_pct=55)
    # barra neon topo
    _rect(s, Inches(0), Inches(0), SLIDE_W, Pt(4), CYAN)
    # titulo
    _txt(
        s,
        Inches(1.5),
        Inches(2.4),
        Inches(10.3),
        Inches(1.2),
        "OBSERVABILIDADE\nEM IA",
        size=52,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    # linha separadora
    _rect(s, Inches(4.5), Inches(3.9), Inches(4.3), Pt(3), CYAN)
    # subtitulo
    _txt(
        s,
        Inches(1.5),
        Inches(4.2),
        Inches(10.3),
        Inches(0.7),
        "Workshop pratico — 1 hora · MLflow GenAI",
        size=22,
        color=CYAN,
        align=PP_ALIGN.CENTER,
    )
    # legenda
    _txt(
        s,
        Inches(1.5),
        Inches(5.0),
        Inches(10.3),
        Inches(0.5),
        "Baseado no projeto ia-observability",
        size=13,
        color=TEXT_MUTED,
        align=PP_ALIGN.CENTER,
    )
    _page(s, 1)

    # ================================================================
    # SLIDE 2 — Section: Por que observabilidade em IA é importante?
    # ================================================================
    s = slide()
    _section_number(s, 1, "Por que observabilidade\nem IA e importante?")
    _page(s, 2)

    # ================================================================
    # SLIDE 3 — Diferença tradicional vs IA
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Diferenca: software tradicional vs. IA",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    # card tradicional
    _card(s, M, Inches(1.4), CONTENT_W, Inches(1.2), CYAN)
    tf = _rich_box(s, Inches(1.2), Inches(1.55), Inches(10.8), Inches(0.9))
    _bullet(
        tf,
        "Software tradicional: mesmo input → mesmo output",
        size=18,
        bold=True,
        color=CYAN,
    )
    _bullet(tf, "Deterministico, reproduzivel, testavel", size=15, color=TEXT_MUTED)

    # card LLM
    _card(s, M, Inches(2.9), CONTENT_W, Inches(3.6), PURPLE)
    tf = _rich_box(s, Inches(1.2), Inches(3.1), Inches(10.8), Inches(3.2))
    _bullet(tf, "Com LLMs isso quebra:", size=18, bold=True, color=PURPLE, sb=Pt(0))
    for item in [
        "Nao-determinismo — mesmo prompt, respostas diferentes",
        "Caixa-preta — nao ve por que o modelo respondeu",
        "Falha silenciosa — alucina com confianca e devolve HTTP 200",
        "Custo variavel — tokens se multiplicam em escala",
        "Pipelines complexos — RAG + tools + LLM: qual etapa falhou?",
    ]:
        _bullet(tf, f"▸  {item}", size=15, color=TEXT_LIGHT, sb=Pt(7))
    _page(s, 3)

    # ================================================================
    # SLIDE 4 — Definição
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "O que e observabilidade em IA?",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    # definicao em destaque
    _rect(s, M, Inches(1.3), CONTENT_W, Inches(1.1), BG_CARD)
    _rect(s, M, Inches(1.3), CONTENT_W, Pt(3), CYAN)
    tf = _rich_box(s, Inches(1.4), Inches(1.5), Inches(10.5), Inches(0.8))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(
        p,
        "Capacidade de entender o estado interno de uma aplicacao de IA\n"
        "atraves de traces, tokens/custo, latencia e qualidade das respostas.",
        size=16,
        color=CYAN,
    )

    # steps numerados
    steps = [
        ("01", "Requisicao", "O input chega na aplicacao LLM", CYAN),
        ("02", "Trace completo", "Inputs, outputs, latencia de cada passo", PURPLE),
        ("03", "Analise", "Tokens, custo, score de qualidade", MINT),
    ]
    for i, (num, title, desc, col) in enumerate(steps):
        y = Inches(2.8) + Inches(1.15) * i
        _rect(s, M, y, Inches(0.6), Inches(0.6), col)
        _txt(
            s,
            Inches(0.85),
            y + Pt(6),
            Inches(0.6),
            Inches(0.6),
            num,
            size=16,
            bold=True,
            color=BG_PRIMARY,
            align=PP_ALIGN.CENTER,
        )
        _txt(
            s,
            Inches(1.7),
            y + Pt(2),
            Inches(4),
            Inches(0.4),
            title,
            size=18,
            bold=True,
            color=WHITE,
        )
        _txt(
            s,
            Inches(1.7),
            y + Inches(0.32),
            Inches(6),
            Inches(0.35),
            desc,
            size=14,
            color=TEXT_MUTED,
        )

    # mensagem final
    _rect(s, M, Inches(6.3), CONTENT_W, Inches(0.65), BG_CARD2)
    _rect(s, M, Inches(6.3), Pt(4), Inches(0.65), PINK)
    _txt(
        s,
        Inches(1.2),
        Inches(6.4),
        Inches(10.5),
        Inches(0.45),
        "Sem observabilidade, melhorar IA e chute. Com ela, vira engenharia.",
        size=15,
        bold=True,
        color=PINK,
    )
    _page(s, 4)

    # ================================================================
    # SLIDE 5 — Section: Caso de uso
    # ================================================================
    s = slide()
    _section_number(s, 2, "Caso de uso e o problema\nque ela resolve")
    _page(s, 5)

    # ================================================================
    # SLIDE 6 — Cenário
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "O cenario",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    # quote card
    _card(s, M, Inches(1.3), CONTENT_W, Inches(1.5), CYAN)
    tf = _rich_box(s, Inches(1.4), Inches(1.5), Inches(10.5), Inches(1.2))
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(
        p,
        '"Um cliente reclamou que o bot deu uma resposta errada as 14h32. '
        "O time nao consegue reproduzir.\nA conta de inferencia triplicou no mes. "
        'Ninguem sabe qual ferramenta esta lenta\nnem qual versao do prompt esta no ar."',
        size=14,
        color=TEXT_LIGHT,
    )
    # perguntas
    _txt(
        s,
        M,
        Inches(3.1),
        Inches(10),
        Inches(0.4),
        "Sem observabilidade, cada pergunta e impossivel de responder:",
        size=16,
        bold=True,
        color=TEXT_LIGHT,
    )
    tf = _rich_box(s, M, Inches(3.6), CONTENT_W, Inches(3.5))
    for prob in [
        "O que o usuario enviou e o que o modelo respondeu?",
        "Qual ferramenta o agente chamou? Ela falhou ou demorou?",
        "Por que a conta de tokens explodiu?",
        "Essa resposta estava certa? Quantas estao erradas?",
        "Qual versao do prompt gerou essa resposta?",
        "Auditar 100% de pagamentos, mas so 10% do resto",
    ]:
        _bullet(tf, f"▸  {prob}", size=15, color=TEXT_MUTED, sb=Pt(5))
    _page(s, 6)

    # ================================================================
    # SLIDE 7 — Tabela problema × solução
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "O que cada problema resolve",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    # tabela
    rows, cols = 7, 2
    tbl_shape = s.shapes.add_table(rows, cols, M, Inches(1.4), CONTENT_W, Inches(5.0))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(7.5)
    tbl.columns[1].width = Inches(4.2)
    # header
    for ci, h in enumerate(["Pergunta do time", "O que resolve"]):
        c = tbl.cell(0, ci)
        c.text = h
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(15)
            p.font.bold = True
            p.font.color.rgb = BG_PRIMARY
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER
        c.fill.solid()
        c.fill.fore_color.rgb = CYAN
    # linhas
    tdata = [
        ["O que o usuario enviou e o que o modelo respondeu?", "Tracing"],
        ["Qual ferramenta o agente chamou? Falhou ou demorou?", "Spans de tool"],
        ["Por que a conta de tokens explodiu?", "Token usage + custo"],
        ["Essa resposta estava certa? Quantas estao erradas?", "Avaliacao / Judges"],
        ["Qual versao do prompt gerou isso?", "Prompt registry"],
        ["Auditar 100% de pagamentos, so 10% do resto", "Sampling"],
    ]
    for ri, row in enumerate(tdata):
        bg_cell = BG_CARD if ri % 2 == 0 else BG_CARD2
        for ci, val in enumerate(row):
            c = tbl.cell(ri + 1, ci)
            c.text = val
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(14)
                p.font.color.rgb = TEXT_LIGHT
                p.font.name = "Calibri"
                p.alignment = PP_ALIGN.LEFT
            c.fill.solid()
            c.fill.fore_color.rgb = bg_cell
            if ci == 1:
                for p in c.text_frame.paragraphs:
                    p.font.bold = True
                    p.font.color.rgb = CYAN
    # bordas via accent lines nas celulas - nao da pra fazer facil no python-pptx
    _page(s, 7)

    # ================================================================
    # SLIDE 8 — Tool lenta no trace
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Como uma tool lenta aparece num trace",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _code(
        s,
        M,
        Inches(1.3),
        CONTENT_W,
        Inches(2.5),
        [
            "@tool",
            "def check_inventory(product: str) -> str:",
            '    """Consulta o estoque disponivel de um produto."""',
            "    # Simula latencia alta seguida de falha",
            "    time.sleep(2.5)",
            "    return f\"ERRO: Timeout ao consultar estoque do produto '{product}'\"",
        ],
        size=15,
    )
    # destaque
    _card(s, M, Inches(4.2), CONTENT_W, Inches(1.0), CYAN)
    _txt(
        s,
        Inches(1.2),
        Inches(4.4),
        Inches(10.5),
        Inches(0.6),
        "No MLflow UI, time.sleep(2.5) vira um span visivel de 2,5s "
        "— o gargalo fica mensuravel, nao um misterio.",
        size=15,
        color=CYAN,
    )
    _page(s, 8)

    # ================================================================
    # SLIDE 9 — Section: Os 4 pilares
    # ================================================================
    s = slide()
    _section_number(s, 3, "Os 4 pilares da\nobservabilidade de LLM")
    _page(s, 9)

    # ================================================================
    # SLIDE 10 — 4 pilares (cards)
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "4 pilares",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    pillars = [
        (
            "TRACING",
            "Registra cada passo da execucao:\ninputs, outputs, latencia",
            CYAN,
        ),
        ("CUSTO", "Quanto cada chamada consome\ne quanto custa", PURPLE),
        ("AVALIACAO", "Mede a qualidade com judges\ne code-based scorers", MINT),
        ("PRODUCAO", "Sampling, feedback humano,\nsessions e users em escala", PINK),
    ]
    w_card = Inches(2.7)
    gap = Inches(0.3)
    total_w = 4 * w_card + 3 * gap
    start_x = (SLIDE_W - total_w) // 2
    for i, (title, desc, col) in enumerate(pillars):
        x = start_x + (w_card + gap) * i
        y = Inches(1.6)
        _rect(s, x, y, w_card, Inches(4.8), BG_CARD)
        _rect(s, x, y, w_card, Pt(4), col)
        _txt(
            s,
            x + Inches(0.2),
            y + Inches(0.4),
            w_card - Inches(0.4),
            Inches(0.5),
            title,
            size=18,
            bold=True,
            color=col,
            align=PP_ALIGN.CENTER,
        )
        _rect(
            s, x + Inches(0.5), y + Inches(1.0), w_card - Inches(1.0), Pt(1), GRAY_DOT
        )
        _txt(
            s,
            x + Inches(0.2),
            y + Inches(1.3),
            w_card - Inches(0.4),
            Inches(2.5),
            desc,
            size=14,
            color=TEXT_LIGHT,
            align=PP_ALIGN.CENTER,
        )
    _page(s, 10)

    # ================================================================
    # SLIDE 11 — Section: Como usar
    # ================================================================
    s = slide()
    _section_number(
        s,
        4,
        "Como usar — demos ao vivo",
        "MLflow GenAI + OpenAI SDK + MLflow AI Gateway",
    )
    _page(s, 11)

    # ================================================================
    # SLIDE 12 — Demo 1: Tracing em 1 linha
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 1 — Tracing em 1 linha",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make tracing  ·  tracing_basics.py",
        size=16,
        color=CYAN,
    )
    _code(
        s,
        M,
        Inches(1.7),
        CONTENT_W,
        Inches(1.6),
        [
            "import mlflow",
            "",
            "mlflow.openai.autolog()   # <- 1 linha instrumenta todas as chamadas",
            "",
            "client = get_client()",
            "response = client.chat.completions.create(",
            "    model=MODEL_NAME, messages=[...],",
            ")",
        ],
        size=14,
    )
    tf = _rich_box(s, M, Inches(3.7), CONTENT_W, Inches(3.0))
    _bullet(
        tf,
        "Captura inputs, outputs, tokens e latencia automaticamente",
        size=17,
        color=CYAN,
        sb=Pt(8),
    )
    _bullet(tf, "Sem nenhuma instrumentacao manual", size=17, color=TEXT_LIGHT)
    _bullet(
        tf, "Abra o trace no MLflow UI e veja tudo pronto", size=17, color=TEXT_LIGHT
    )
    _bullet(
        tf,
        "Gotcha: flush_trace_async_logging() antes de search_traces()",
        size=15,
        color=PINK,
        sb=Pt(16),
    )
    _page(s, 12)

    # ================================================================
    # SLIDE 13 — Demo 2: Spans aninhados
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 2 — Pipelines complexos: spans aninhados",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make tracing  ·  tracing_basics.py",
        size=16,
        color=CYAN,
    )
    _code(
        s,
        M,
        Inches(1.7),
        CONTENT_W,
        Inches(2.8),
        [
            "@mlflow.trace",
            "def demo_rag_pipeline(question: str) -> str:",
            "    context = retrieve_context(question)        # span filho",
            "    answer = generate_answer(question, context) # span filho",
            "    return answer",
            "",
            '@mlflow.trace(span_type="RETRIEVER")',
            "def retrieve_context(question: str) -> str:",
            "    ...  # busca em vector store",
            "",
            '@mlflow.trace(span_type="LLM")',
            "def generate_answer(question: str, context: str) -> str:",
            "    ...  # chamada ao LLM com o contexto",
        ],
        size=13,
    )
    _card(s, M, Inches(4.9), CONTENT_W, Inches(1.0), CYAN)
    _txt(
        s,
        Inches(1.2),
        Inches(5.1),
        Inches(10.5),
        Inches(0.6),
        "Resposta errada? O trace mostra: foi o retrieval (contexto ruim) "
        "ou a geracao (modelo)?",
        size=15,
        color=CYAN,
    )
    _page(s, 13)

    # ================================================================
    # SLIDE 14 — Demo 3: Tokens + cost
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 3 — Quanto isso custa?",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make tokens  ·  token_usage.py",
        size=16,
        color=CYAN,
    )
    # dois cards lado a lado
    _card(s, M, Inches(1.7), Inches(5.6), Inches(2.0), MINT)
    tf = _rich_box(s, Inches(1.4), Inches(1.9), Inches(5.2), Inches(1.6))
    _bullet(tf, "MLflow calcula custo", size=16, bold=True, color=MINT, sb=Pt(0))
    _bullet(tf, "automatico para OpenAI,", size=16, bold=True, color=MINT)
    _bullet(tf, "Anthropic e outros", size=16, bold=True, color=MINT)

    _card(s, Inches(6.8), Inches(1.7), Inches(5.6), Inches(2.0), PINK)
    tf = _rich_box(s, Inches(7.2), Inches(1.9), Inches(5.0), Inches(1.6))
    _bullet(tf, "Para self-hosted:", size=16, bold=True, color=PINK, sb=Pt(0))
    _bullet(tf, "atribua custo manualmente", size=16, bold=True, color=PINK)
    _bullet(tf, "no span", size=16, bold=True, color=PINK)

    _code(
        s,
        M,
        Inches(4.1),
        CONTENT_W,
        Inches(1.8),
        [
            'span.set_attribute("mlflow.llm.cost", {',
            '    "input_cost": input_tokens * CUSTOM_INPUT_COST,',
            '    "output_cost": output_tokens * CUSTOM_OUTPUT_COST,',
            "})",
        ],
        size=14,
    )
    _txt(
        s,
        M,
        Inches(6.2),
        Inches(11),
        Inches(0.4),
        "Veja Cost Breakdown e Token Usage por trace no MLflow UI",
        size=15,
        bold=True,
        color=CYAN,
    )
    _page(s, 14)

    # ================================================================
    # SLIDE 15 — Demo 4: Judges
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 4 — A resposta estava certa?",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make judges  ·  judges.py",
        size=16,
        color=CYAN,
    )
    # LLM Judge card
    _card(s, M, Inches(1.7), Inches(5.6), Inches(4.5), CYAN)
    _txt(
        s,
        Inches(1.2),
        Inches(1.9),
        Inches(5.2),
        Inches(0.4),
        "LLM Judge",
        size=20,
        bold=True,
        color=CYAN,
    )
    _txt(
        s,
        Inches(1.2),
        Inches(2.35),
        Inches(5.2),
        Inches(0.3),
        "Modelo julga a resposta em linguagem natural",
        size=13,
        color=TEXT_MUTED,
    )
    _code(
        s,
        Inches(1.2),
        Inches(2.8),
        Inches(5.0),
        Inches(1.8),
        [
            "Guidelines(",
            '    name="technical_accuracy",',
            '    guidelines=("A resposta deve ser',
            '        tecnica e precisa."),',
            "    model=JUDGE_MODEL,",
            ")",
        ],
        size=12,
    )
    # Code-based card
    _card(s, Inches(6.8), Inches(1.7), Inches(5.6), Inches(4.5), MINT)
    _txt(
        s,
        Inches(7.2),
        Inches(1.9),
        Inches(5.0),
        Inches(0.4),
        "Code-based Scorer",
        size=20,
        bold=True,
        color=MINT,
    )
    _txt(
        s,
        Inches(7.2),
        Inches(2.35),
        Inches(5.0),
        Inches(0.3),
        "Regra deterministica sem custo de LLM",
        size=13,
        color=TEXT_MUTED,
    )
    _code(
        s,
        Inches(7.2),
        Inches(2.8),
        Inches(5.0),
        Inches(1.8),
        [
            "@scorer",
            "def no_hallucination(inputs, outputs):",
            "    red_flags = [...],",
            "    if any(f in outputs for f in red_flags):",
            "        return Feedback(value=False)",
            "    return Feedback(value=True)",
        ],
        size=12,
    )
    # barra dica
    _rect(s, M, Inches(6.5), CONTENT_W, Inches(0.5), BG_CARD2)
    _rect(s, M, Inches(6.5), Pt(4), Inches(0.5), PURPLE)
    _txt(
        s,
        Inches(1.2),
        Inches(6.55),
        Inches(10.5),
        Inches(0.4),
        "Combine os dois: LLM judges para qualidade subjetiva + "
        "scorers para regras objetivas sem custo.",
        size=14,
        color=PURPLE,
    )
    _page(s, 15)

    # ================================================================
    # SLIDE 16 — Demo 5: Streaming + span manual
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 5 — Agente real com streaming",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make langchain-agent  ·  langchain_agent.py",
        size=16,
        color=CYAN,
    )
    # badge producao
    _rect(s, M, Inches(1.6), Inches(2.8), Inches(0.4), BG_CARD2)
    _rect(s, M, Inches(1.6), Pt(4), Inches(0.4), PINK)
    _txt(
        s,
        M,
        Inches(1.68),
        Inches(2.4),
        Inches(0.4),
        "PROD",
        size=9,
        bold=True,
        color=PINK,
        align=PP_ALIGN.CENTER,
    )
    _txt(
        s,
        Inches(1.7),
        Inches(1.7),
        Inches(10),
        Inches(0.4),
        "Mesmo padrao do backend de producao keepee-rag RAG",
        size=16,
        bold=True,
        color=PINK,
    )
    # topicos
    tf = _rich_box(s, M, Inches(2.4), CONTENT_W, Inches(4.5))
    for item in [
        "agent.astream() — streaming token a token em tempo real",
        "Span manual SpanType.AGENT com inputs/outputs explicitos",
        "Session em dict (reconstrucao manual do historico)",
        "get_stream_writer() para logs de progresso das tools",
        "Eventos JSON newline-delimited via make_event()",
        "trace_id capturado do span e exposto para feedback",
        "User/session vinculados ao trace ANTES do stream",
        "Tags (provider, model_name) setadas apos o stream",
    ]:
        color = (
            CYAN if "astream" in item else PURPLE if "manual" in item else TEXT_LIGHT
        )
        _bullet(tf, f"▸  {item}", size=15, color=color, sb=Pt(8))
    _page(s, 16)

    # ================================================================
    # SLIDE 17 — Demo 5: autolog vs manual (tabela)
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "autolog vs. Manual Span",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    rows, cols = 7, 3
    tbl_shape = s.shapes.add_table(rows, cols, M, Inches(1.4), CONTENT_W, Inches(5.0))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(4.8)
    tbl.columns[2].width = Inches(4.7)
    for ci, h in enumerate(["Aspecto", "autolog()", "Producao (manual span)"]):
        c = tbl.cell(0, ci)
        c.text = h
        for p in c.text_frame.paragraphs:
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = BG_PRIMARY
            p.font.name = "Calibri"
            p.alignment = PP_ALIGN.CENTER
        c.fill.solid()
        c.fill.fore_color.rgb = CYAN
    tdata = [
        ["Streaming", "invoke() bloqueante", "astream() token a token"],
        ["Session", "MemorySaver (checkpointer)", "Dict (reconstrucao manual)"],
        ["Controle", "Automatico (generico)", "Inputs/outputs explicitos"],
        ["Progresso tools", "So o resultado final", "Logs via get_stream_writer()"],
        ["Eventos", "Apenas no trace", "JSON newline-delimited (SSE)"],
        ["trace_id", "get_last_active_trace_id()", "Capturado do span manual"],
    ]
    for ri, row in enumerate(tdata):
        bg_cell = BG_CARD if ri % 2 == 0 else BG_CARD2
        for ci, val in enumerate(row):
            c = tbl.cell(ri + 1, ci)
            c.text = val
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.color.rgb = TEXT_LIGHT
                p.font.name = "Calibri"
            c.fill.solid()
            c.fill.fore_color.rgb = bg_cell
            if ci == 0:
                for p in c.text_frame.paragraphs:
                    p.font.bold = True
                    p.font.color.rgb = CYAN
            if ci == 2:
                for p in c.text_frame.paragraphs:
                    p.font.color.rgb = MINT
    _page(s, 17)

    # ================================================================
    # SLIDE 18 — Demo 6: Feedback + Sampling
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Demo 6 — Operando em producao",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _txt(
        s,
        M,
        Inches(1.1),
        Inches(11),
        Inches(0.4),
        "make monitoring  ·  production_monitoring.py",
        size=16,
        color=CYAN,
    )
    # Sampling
    _card(s, M, Inches(1.7), Inches(5.6), Inches(2.5), PURPLE)
    _txt(
        s,
        Inches(1.2),
        Inches(1.9),
        Inches(5.0),
        Inches(0.4),
        "Sampling por criticidade",
        size=18,
        bold=True,
        color=PURPLE,
    )
    _txt(
        s,
        Inches(1.2),
        Inches(2.3),
        Inches(5.0),
        Inches(0.3),
        "Nao traceie 100% — storage e custo importam",
        size=13,
        color=TEXT_MUTED,
    )
    _code(
        s,
        Inches(1.2),
        Inches(2.7),
        Inches(5.0),
        Inches(1.2),
        [
            "@mlflow.trace(sampling_ratio_override=1.0)  # 100% critico",
            "@mlflow.trace(sampling_ratio_override=0.1)  # 10% volume",
            "def agent_call(...)",
        ],
        size=12,
    )
    # Feedback
    _card(s, Inches(6.8), Inches(1.7), Inches(5.6), Inches(2.5), CYAN)
    _txt(
        s,
        Inches(7.2),
        Inches(1.9),
        Inches(5.0),
        Inches(0.4),
        "Feedback humano no trace",
        size=18,
        bold=True,
        color=CYAN,
    )
    _code(
        s,
        Inches(7.2),
        Inches(2.4),
        Inches(5.0),
        Inches(1.5),
        [
            "trace_id = get_last_active_trace_id()",
            "mlflow.log_feedback(",
            "    trace_id=trace_id,",
            '    name="user_rating",',
            "    value=True,  # like/dislike",
            '    rationale="texto do usuario",',
            ")",
        ],
        size=12,
    )
    # gotcha
    _rect(s, M, Inches(4.8), CONTENT_W, Inches(0.5), BG_CARD2)
    _rect(s, M, Inches(4.8), Pt(4), Inches(0.5), PINK)
    _txt(
        s,
        Inches(1.2),
        Inches(4.87),
        Inches(11.3),
        Inches(0.35),
        "flush_trace_async_logging() antes de search_traces()!",
        size=15,
        bold=True,
        color=PINK,
    )
    _page(s, 18)

    # ================================================================
    # SLIDE 19 — Section: Fechamento
    # ================================================================
    s = slide()
    _section_number(s, 5, "Fechamento")
    _page(s, 19)

    # ================================================================
    # SLIDE 20 — Ciclo virtuoso
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "O ciclo virtuoso da observabilidade",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    flow = [
        ("TRACING", "Vejo o que\nacontece", CYAN),
        ("AVALIACAO", "Meco a\nqualidade", PURPLE),
        ("OTIMIZACAO", "Melhoro prompts\ne tools", MINT),
    ]
    w_card = Inches(3.2)
    gap = Inches(0.6)
    total_w = 3 * w_card + 2 * gap
    start_x = (SLIDE_W - total_w) // 2
    for i, (title, desc, col) in enumerate(flow):
        x = start_x + (w_card + gap) * i
        y = Inches(1.8)
        _rect(s, x, y, w_card, Inches(3.2), BG_CARD)
        _rect(s, x, y, w_card, Pt(4), col)
        _txt(
            s,
            x,
            y + Inches(0.5),
            w_card,
            Inches(0.5),
            title,
            size=18,
            bold=True,
            color=col,
            align=PP_ALIGN.CENTER,
        )
        _txt(
            s,
            x,
            y + Inches(1.2),
            w_card,
            Inches(1.5),
            desc,
            size=16,
            color=TEXT_LIGHT,
            align=PP_ALIGN.CENTER,
        )
        # seta entre cards (exceto ultimo)
        if i < 2:
            ax = x + w_card + Pt(6)
            _txt(
                s,
                ax,
                y + Inches(1.2),
                Inches(0.5),
                Inches(0.5),
                ">",
                size=28,
                bold=True,
                color=col,
                align=PP_ALIGN.CENTER,
            )
    # seta de retorno
    _txt(
        s,
        Inches(5.0),
        Inches(5.3),
        Inches(3.3),
        Inches(0.5),
        "^  fecha o ciclo",
        size=14,
        bold=True,
        color=CYAN,
        align=PP_ALIGN.CENTER,
    )
    # mensagem final
    _rect(s, M, Inches(6.2), CONTENT_W, Inches(0.65), BG_CARD2)
    _rect(s, M, Inches(6.2), Pt(4), Inches(0.65), CYAN)
    _txt(
        s,
        Inches(1.2),
        Inches(6.3),
        Inches(10.5),
        Inches(0.45),
        'Observabilidade nao e "ver logs" — e o que fecha o ciclo '
        "entre observar, medir e melhorar sua IA.",
        size=15,
        bold=True,
        color=CYAN,
    )
    _page(s, 20)

    # ================================================================
    # SLIDE 21 — Checklist
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Checklist",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    checks = [
        "Ligue auto-tracing (mlflow.openai.autolog)",
        "Use spans aninhados em pipelines (RAG, agentes)",
        "Atribua custo manualmente se self-hosted",
        "Combine LLM judges + code-based scorers",
        "Vincule user_id e session_id aos traces",
        "Producao: sampling por criticidade + feedback humano",
        "flush_trace_async_logging() antes de ler traces",
    ]
    tf = _rich_box(s, M, Inches(1.4), CONTENT_W, Inches(5.5))
    for item in checks:
        _bullet(tf, f"[ ]  {item}", size=18, color=TEXT_LIGHT, sb=Pt(12))
    _page(s, 21)

    # ================================================================
    # SLIDE 22 — Comandos
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(0.5),
        Inches(11),
        Inches(0.6),
        "Comandos para explorar",
        size=TITLE_SIZE,
        bold=True,
        color=WHITE,
    )
    _code(
        s,
        M,
        Inches(1.3),
        Inches(5.5),
        Inches(5.5),
        [
            "# Setup",
            "uv sync",
            "",
            "# Demos",
            "make tracing            # 01",
            "make tokens             # 02",
            "make judges             # 05",
            "make langchain-agent    # 11",
            "make monitoring         # 07",
            "",
            "make help    # lista todas",
        ],
        size=15,
    )
    # info box
    _card(s, Inches(6.8), Inches(1.3), Inches(5.5), Inches(2.8), CYAN)
    tf = _rich_box(s, Inches(7.2), Inches(1.5), Inches(4.8), Inches(2.4))
    _bullet(tf, "MLflow UI", size=18, bold=True, color=CYAN, sb=Pt(0))
    _bullet(tf, "http://localhost:5000", size=15, color=TEXT_LIGHT)
    _bullet(tf, "", size=8)
    _bullet(tf, "MLflow AI Gateway", size=18, bold=True, color=CYAN)
    _bullet(tf, "http://localhost:5000/gateway/mlflow/v1", size=15, color=TEXT_LIGHT)
    _bullet(tf, "", size=8)
    _bullet(tf, "Docker Compose:", size=18, bold=True, color=CYAN)
    _bullet(tf, "podman compose up -d", size=15, color=TEXT_LIGHT)
    # experiments
    _card(s, Inches(6.8), Inches(4.5), Inches(5.5), Inches(2.3), PURPLE)
    _txt(
        s,
        Inches(7.2),
        Inches(4.6),
        Inches(4.8),
        Inches(0.3),
        "Experiments:",
        size=16,
        bold=True,
        color=PURPLE,
    )
    _code(
        s,
        Inches(7.2),
        Inches(5.0),
        Inches(4.8),
        Inches(1.5),
        [
            "01 - tracing_basics",
            "02 - token_usage",
            "05 - judges",
            "11 - langchain_agent",
            "07 - production_monitoring",
        ],
        size=14,
    )
    _page(s, 22)

    # ================================================================
    # SLIDE 23 — Obrigado / Referencias
    # ================================================================
    s = slide()
    _set_bg(s, BG_PRIMARY)
    _accent_bar(s, Inches(0))
    _txt(
        s,
        M,
        Inches(2.0),
        Inches(11),
        Inches(1),
        "Obrigado!",
        size=48,
        bold=True,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    _rect(s, Inches(4.5), Inches(3.2), Inches(4.3), Pt(3), CYAN)
    _txt(
        s,
        M,
        Inches(3.6),
        Inches(11),
        Inches(0.4),
        "Referencias",
        size=18,
        bold=True,
        color=CYAN,
        align=PP_ALIGN.CENTER,
    )
    tf = _rich_box(s, Inches(2), Inches(4.2), Inches(9.3), Inches(2.8))
    for ref in [
        "MLflow GenAI Docs  —  mlflow.org/docs/latest/genai",
        "Tracing Quickstart  —  mlflow.org/docs/latest/genai/tracing/quickstart",
        "Evaluation & Monitoring  —  mlflow.org/docs/latest/genai/eval-monitor",
        "LLM Judges / Scorers  —  mlflow.org/docs/latest/genai/eval-monitor/scorers",
        "Production Monitoring  —  mlflow.org/docs/latest/genai/tracing/prod-tracing",
    ]:
        _bullet(tf, ref, size=15, color=TEXT_LIGHT, sb=Pt(10))
    _page(s, 23)

    # ================================================================
    # Salvar
    # ================================================================
    path = "docs/workshop-observabilidade-ia.pptx"
    prs.save(path)
    print(f"Salvo: {path}")


if __name__ == "__main__":
    build()
