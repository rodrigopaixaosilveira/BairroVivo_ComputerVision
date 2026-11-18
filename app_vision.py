"""BairroVivo Vision — detecção em tempo real das fotos de ocorrências que chegam ao BairroVivo.

Recebe um fluxo de imagens (um vídeo; na demo, fotos de moradores uma atrás da outra). Cada problema detectado é
rastreado (ByteTrack); quando aparece em pelo menos --min-frames frames, vira UMA ocorrência sugerida (categoria,
prioridade, momento e foto-evidência). Pessoas têm a cabeça borrada na exibição (LGPD).

Uso:
    python app_vision.py --video videos/demo_fotos_moradores.mp4 --modelo modelos/bairrovivo_vision.pt --salvar
Teclas: espaço = pausa, q ou Esc = sair.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vision_lib import CORES_PRIORIDADE, categoria, imwrite  # noqa: E402

ALTURA, PAINEL_W = 720, 420


def fonte(t, negrito=False):
    for nome in (['segoeuib.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf'] if negrito else
                 ['segoeui.ttf', 'arial.ttf', 'DejaVuSans.ttf']):
        try:
            return ImageFont.truetype(nome, t)
        except OSError:
            continue
    return ImageFont.load_default()


class Registro:
    """Transforma detecções rastreadas em ocorrências (uma por objeto)."""

    def __init__(self, min_frames=5, pasta_evid=None):
        self.min_frames, self.pasta = min_frames, pasta_evid
        self.vistos = Counter()
        self.ocorrencias = {}           # tid -> dict

    def atualizar(self, t, frame, dets):
        """dets: [(nome_classe, conf, caixa_xyxy, tid)]"""
        for nome, conf, caixa, tid in dets:
            if tid is None:
                continue
            self.vistos[tid] += 1
            oc = self.ocorrencias.get(tid)
            if oc is None and self.vistos[tid] >= self.min_frames:
                c = categoria(nome)
                oc = dict(id=len(self.ocorrencias) + 1, tid=tid, t=round(t, 1), conf=conf, **c, evidencia='')
                self.ocorrencias[tid] = oc
            if oc is not None and conf >= oc['conf']:
                oc['conf'] = conf
                if self.pasta is not None:
                    x0, y0, x1, y1 = map(int, caixa)
                    m = 20
                    rec = frame[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]
                    if rec.size:
                        p = Path(self.pasta) / f"ocorrencia_{oc['id']:03d}.jpg"
                        imwrite(p, rec)
                        oc['evidencia'] = p.name

    def por_categoria(self):
        return Counter(o['categoria'] for o in self.ocorrencias.values())


class Painel:
    def __init__(self):
        self.f_tit, self.f_med, self.f_txt, self.f_peq, self.f_big = fonte(22, True), fonte(19, True), fonte(17), fonte(14), fonte(46, True)

    def desenhar(self, t, dets_frame, reg: Registro, fps, modelo):
        img = Image.new('RGB', (PAINEL_W, ALTURA), (24, 26, 32))
        d = ImageDraw.Draw(img)
        x, w = 20, PAINEL_W
        d.text((x, 14), 'BairroVivo · Vision', font=self.f_tit, fill=(235, 235, 240))
        mm, ss = divmod(int(t), 60)
        d.text((x, 44), f'Triagem automática de ocorrências · {mm:02d}:{ss:02d}', font=self.f_peq, fill=(150, 155, 165))

        n = len(reg.ocorrencias)
        d.text((x, 72), str(n), font=self.f_big, fill=(240, 240, 245))
        d.text((x + d.textlength(str(n), font=self.f_big) + 12, 88), 'ocorrências sugeridas\nnesta sessão',
               font=self.f_peq, fill=(150, 155, 165))

        y = 140
        d.text((x, y), 'Por categoria', font=self.f_med, fill=(235, 235, 240)); y += 28
        cats = reg.por_categoria()
        if not cats:
            d.text((x, y), 'nenhuma ainda', font=self.f_txt, fill=(130, 135, 145)); y += 24
        maxc = max(cats.values()) if cats else 1
        for cat, q in cats.most_common(6):
            prio = next(o['prioridade'] for o in reg.ocorrencias.values() if o['categoria'] == cat)
            d.text((x, y), cat, font=self.f_txt, fill=(210, 210, 218))
            d.text((w - x, y), str(q), font=self.f_txt, fill=(210, 210, 218), anchor='ra')
            d.rounded_rectangle((x, y + 22, w - x, y + 30), radius=4, fill=(55, 58, 68))
            d.rounded_rectangle((x, y + 22, x + (w - 2 * x) * q / maxc, y + 30), radius=4, fill=CORES_PRIORIDADE[prio])
            y += 40

        y += 6
        d.text((x, y), 'Neste frame', font=self.f_med, fill=(235, 235, 240)); y += 28
        if not dets_frame:
            d.text((x, y), '—', font=self.f_txt, fill=(130, 135, 145)); y += 24
        for nome, conf, _, tid in sorted(dets_frame, key=lambda a: -a[1])[:5]:
            c = categoria(nome)
            d.rounded_rectangle((x, y + 3, x + 12, y + 15), radius=3, fill=CORES_PRIORIDADE[c['prioridade']])
            d.text((x + 20, y), f"{c['nome_pt']}", font=self.f_txt, fill=(210, 210, 218))
            d.text((w - x, y), f'{conf * 100:.0f}%', font=self.f_txt, fill=(210, 210, 218), anchor='ra')
            y += 24

        y = max(y + 10, ALTURA - 230)
        d.text((x, y), 'Últimas ocorrências', font=self.f_med, fill=(235, 235, 240)); y += 26
        for o in list(reg.ocorrencias.values())[-4:][::-1]:
            m2, s2 = divmod(int(o['t']), 60)
            d.text((x, y), f"#{o['id']:03d}  {m2:02d}:{s2:02d}  {o['nome_pt']}", font=self.f_peq, fill=(200, 200, 210))
            d.text((w - x, y), o['prioridade'], font=self.f_peq, fill=CORES_PRIORIDADE[o['prioridade']], anchor='ra')
            y += 20

        y = ALTURA - 76
        d.text((x, y), 'Prioridade:', font=self.f_peq, fill=(170, 175, 185))
        xx = x + 84
        for prio in ('alta', 'média', 'baixa'):
            d.rectangle((xx, y + 3, xx + 10, y + 13), fill=CORES_PRIORIDADE[prio])
            d.text((xx + 16, y), prio, font=self.f_peq, fill=(170, 175, 185))
            xx += 80
        d.text((x, ALTURA - 50), 'Sugestão automática: o operador confirma.', font=self.f_peq, fill=(150, 155, 165))
        rod = f'{fps:4.1f} fps · {modelo}'
        d.text((x, ALTURA - 28), rod[:60], font=self.f_peq, fill=(110, 115, 125))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def borrar_cabecas(frame, caixas_pessoas):
    for x0, y0, x1, y1 in caixas_pessoas:
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        hh = max(8, int((y1 - y0) * 0.22))
        reg = frame[max(0, y0):y0 + hh, max(0, x0):x1]
        if reg.size:
            k = max(15, (min(reg.shape[:2]) // 2) * 2 + 1)
            frame[max(0, y0):y0 + hh, max(0, x0):x1] = cv2.GaussianBlur(reg, (k, k), 0)


_FONTE_CAIXA = None


def desenhar_caixas(frame, dets, reg: Registro):
    """Caixas com o nome em português (acentos via PIL)."""
    global _FONTE_CAIXA
    if not dets:
        return
    if _FONTE_CAIXA is None:
        _FONTE_CAIXA = fonte(18, True)
    for nome, conf, caixa, tid in dets:
        x0, y0, x1, y1 = map(int, caixa)
        cv2.rectangle(frame, (x0, y0), (x1, y1), CORES_PRIORIDADE[categoria(nome)['prioridade']][::-1], 3)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for nome, conf, caixa, tid in dets:
        c = categoria(nome)
        x0, y0 = int(caixa[0]), int(caixa[1])
        oc = reg.ocorrencias.get(tid)
        txt = (f"#{oc['id']} " if oc else '') + f"{c['nome_pt']} {conf * 100:.0f}%"
        l, t, r, b = d.textbbox((0, 0), txt, font=_FONTE_CAIXA)
        yy = max(0, y0 - (b - t) - 8)
        d.rectangle((x0, yy, x0 + (r - l) + 8, yy + (b - t) + 8), fill=CORES_PRIORIDADE[c['prioridade']])
        d.text((x0 + 4, yy + 2 - t), txt, font=_FONTE_CAIXA, fill=(20, 20, 20))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--modelo', required=True)
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--min-frames', type=int, default=5, help='frames seguidos para virar ocorrência')
    ap.add_argument('--inicio', type=float, default=0.0)
    ap.add_argument('--fim', type=float, default=None)
    ap.add_argument('--salvar', action='store_true')
    ap.add_argument('--sem-janela', action='store_true')
    ap.add_argument('--sem-lgpd', action='store_true', help='não borrar cabeças (não recomendado)')
    ap.add_argument('--sufixo', default='')
    a = ap.parse_args(argv)

    from ultralytics import YOLO
    base = Path(__file__).resolve().parent
    video = Path(a.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f'Não abri {video}. Caminhos com acento não funcionam no OpenCV do Windows.')
    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30
    if a.inicio:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.inicio * 1000)
    modelo = YOLO(a.modelo)
    pessoas = None if a.sem_lgpd else YOLO(str(base / 'yolo11n.pt') if (base / 'yolo11n.pt').exists() else 'yolo11n.pt')

    saidas = base / 'saidas'
    saidas.mkdir(exist_ok=True)
    evid = saidas / f'evidencias_{video.stem}{a.sufixo}'
    reg, painel = Registro(a.min_frames, evid if a.salvar else None), Painel()
    writer = None
    t_ant, fps_ema = time.time(), 0.0
    pausado = False
    while True:
        if not pausado:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if a.fim is not None and t > a.fim:
                break
            H, W = frame.shape[:2]
            if H != ALTURA:
                frame = cv2.resize(frame, (int(W * ALTURA / H), ALTURA))
            # detecta no frame ORIGINAL (o borrão das cabeças confundia o modelo) e só depois borra para exibir/salvar
            r = modelo.track(frame, conf=a.conf, persist=True, tracker='bytetrack.yaml', verbose=False)[0]
            if pessoas is not None:
                rp = pessoas.predict(frame, classes=[0], conf=0.3, verbose=False)[0]
                borrar_cabecas(frame, rp.boxes.xyxy.tolist())
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            dets = [(r.names[int(c)], float(p), b, tid) for c, p, b, tid in
                    zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), ids)]
            reg.atualizar(t, frame, dets)
            vis = frame.copy()
            desenhar_caixas(vis, dets, reg)
            agora = time.time()
            fps_ema = 0.9 * fps_ema + 0.1 * (1 / max(1e-6, agora - t_ant)) if fps_ema else 1 / max(1e-6, agora - t_ant)
            t_ant = agora
            tela = np.hstack([vis, painel.desenhar(t, dets, reg, fps_ema, Path(a.modelo).name)])
            if a.salvar and writer is None:
                for cc in ('mp4v', 'avc1'):
                    writer = cv2.VideoWriter(str(saidas / f'demo_vision_{video.stem}{a.sufixo}.mp4'),
                                             cv2.VideoWriter_fourcc(*cc), fps_v, (tela.shape[1], tela.shape[0]))
                    if writer.isOpened():
                        break
            if writer is not None:
                writer.write(tela)
        if not a.sem_janela:
            cv2.imshow('BairroVivo Vision', tela)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            if k == ord(' '):
                pausado = not pausado
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    csv_p = saidas / f'ocorrencias_{video.stem}{a.sufixo}.csv'
    campos = ['id', 't', 'nome_pt', 'categoria', 'prioridade', 'conf', 'classe', 'evidencia']
    with open(csv_p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction='ignore')
        w.writeheader()
        for o in reg.ocorrencias.values():
            w.writerow({**o, 'conf': round(o['conf'], 3)})
    print(f'{len(reg.ocorrencias)} ocorrências sugeridas:', dict(reg.por_categoria()))
    print('Lista em', csv_p)


if __name__ == '__main__':
    main()
