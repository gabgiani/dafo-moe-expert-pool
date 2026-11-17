#!/usr/bin/env python3
"""Corpus de tareas SIN herramientas de atención al cliente para calibrar y para validar un recorte de expertos.

Los textos de las tareas son sintéticos (triaje de mensajes F, prelectura
de correo H, respuesta final desde notas E). Los prompts de sistema reales no se conocen palabra por palabra: aquí se
usan descripciones equivalentes del formato de salida, así que es una APROXIMACIÓN a la tarea real.

Dos conjuntos con mensajes DISTINTOS (calibración y validación): lo que se usa para elegir expertos no se usa para evaluar.

  moe-corpus.py run --base http://127.0.0.1:18099/v1 --model g --set cal|val [--log fichero.jsonl] [--label TEXTO] [--reps 3]
'cal' solo manda las peticiones (sirve para grabar la traza de expertos); 'val' además las puntúa.
"""
import argparse, json, re, sys, time, urllib.request

TRIAGE = ('You triage a message from a customer of a manufacturing company. Answer with ONE JSON object and nothing else: '
          '{"route": "reply|request|new_order|data", "reply": string, "tables": [string], "summary": string, "order": string, "topic": string}. '
          'route=reply for greetings, thanks, off-topic or unclear messages (write a short polite reply in "reply"); '
          'route=data when the answer needs the customer\'s records (list the tables among orders, order_lines, deliveries, invoices, payments in "tables"); '
          'route=request when the customer asks the company to do something such as cancel or change an order (one-sentence "summary", the order number in "order", a short "topic", nothing in "reply"); '
          'route=new_order when the customer wants to place an order. Answer in English.')
MAIL = ('You read one incoming email of a manufacturing company and answer with ONE JSON object and nothing else: '
        '{"kind": "customer_order|customer_rfq|supplier_reply|enquiry|other", "summary": string (one English sentence), "ask": string, "customer": string, '
        '"refs": [string] (every order, PO, quotation or invoice number mentioned), "action": "answer_from_360|open_form|ask_sender|none", "form": string}. '
        'Never invent numbers.')
FINAL = ('Notes taken from the customer\'s SAP data, one section per area:\n{notes}\n\nNow write the final answer. Use ONLY these notes: do not call any tool '
         'and do not mention steps or notes. Write in British English, give amounts with their currency code, dates like 31 Aug 2026. Question: {question}')

# (id, sistema, usuario, comprobación solo para el conjunto de validación)
CAL = [
    ("F1", TRIAGE, "hi", lambda d: d.get("route") == "reply"),
    ("F2", TRIAGE, "any unpaid invoices?", lambda d: d.get("route") == "data" and "invoices" in d.get("tables", [])),
    ("F3", TRIAGE, "when will order 161690 arrive?", lambda d: d.get("route") == "data"),
    ("F4", TRIAGE, "please cancel order 161854", lambda d: d.get("route") == "request" and "161854" in str(d.get("order", ""))),
    ("F5", TRIAGE, "can I order the same as last time?", lambda d: d.get("route") == "new_order"),
    ("F6", TRIAGE, "who won the football?", lambda d: d.get("route") == "reply"),
    ("F7", TRIAGE, "what about the other one?", lambda d: d.get("route") == "reply"),
    ("H1", MAIL, "Subject: PO 0040-138285-1\nFrom: buyer@acme-industrial.example\nBody: Please find attached our purchase order 0040-138285-1 for 200 pcs of item 109295.\nAttachments: PO_0040-138285-1.pdf",
     lambda d: d.get("action") == "open_form" and "0040-138285-1" in " ".join(d.get("refs", []))),
    ("H2", MAIL, "Subject: Where is my order?\nFrom: purchasing@acme-industrial.example\nBody: Where is my order 161690? It was due last week.\nAttachments: none",
     lambda d: d.get("action") == "answer_from_360" and "161690" in " ".join(d.get("refs", []))),
    ("H3", MAIL, "Subject: Quote request\nFrom: j.smith@northwind.example\nBody: We need a quote for 200 m of item X-4410, delivery to Hamburg.\nAttachments: none",
     lambda d: d.get("action") == "open_form"),
    ("E1", None, FINAL.format(notes="## Customer Invoices\n- Total invoiced: 1,738,122.91 GBP; still to be paid: 912,128.68 GBP.\n- 9 invoices overdue; the oldest is 90012, due 30 Jun 2026, 25 days overdue, open 104,500.00 GBP.\n## Sales Orders\n- 11 open orders worth 4,430,937.35 GBP.",
                              question="Summarise the billing position and any overdue balance."),
     None),
]
VAL = [
    ("F8", TRIAGE, "thanks, that's all", lambda d: d.get("route") == "reply"),
    ("F9", TRIAGE, "how much do I owe you in total?", lambda d: d.get("route") == "data"),
    ("F10", TRIAGE, "I need to change the delivery date of order 162003 to next week", lambda d: d.get("route") == "request" and "162003" in str(d.get("order", ""))),
    ("F11", TRIAGE, "I want to place a new order for 40 of item 109295", lambda d: d.get("route") == "new_order"),
    ("F12", TRIAGE, "what's the capital of France", lambda d: d.get("route") == "reply"),
    ("H4", MAIL, "Subject: Your RFQ 7700123\nFrom: sales@steelworks.example\nBody: Attached our prices for your request 7700123: 4.20 EUR per kg, 2 weeks delivery.\nAttachments: quote_7700123.pdf",
     lambda d: d.get("kind") == "supplier_reply" and "7700123" in " ".join(d.get("refs", []))),
    ("H5", MAIL, "Subject: Invoice 90044 reminder\nFrom: accounts@globex.example\nBody: Please confirm payment of invoice 90044, due last Friday.\nAttachments: none",
     lambda d: "90044" in " ".join(d.get("refs", []))),
    ("E2", None, FINAL.format(notes="## Sales Orders\n- 11 open orders worth 4,430,937.35 GBP.\n- Order 161690 (customer PO 4500123) is due 15 Nov 2026 and is partly delivered: 120 of 200 pieces.\n## Deliveries\n- 1 delivery in transit, expected 12 Nov 2026.",
                              question="What is the fulfilment progress for this customer's orders?"),
     lambda text: "161690" in text and "120" in text and "notes" not in text.lower()),
]


def parse_json(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def ask(base, model, system, user, seed, max_tokens, thinking=False):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": messages, "temperature": 0.3, "seed": seed, "max_tokens": max_tokens}
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=900) as response:
        data = json.load(response)
    timings = data.get("timings") or {}
    return data["choices"][0], {"gen": round(timings.get("predicted_per_second") or 0, 1), "tokens": timings.get("predicted_n") or 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--set", choices=["cal", "val"], required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--log")
    parser.add_argument("--thinking", action="store_true", help="con razonamiento (plantilla por defecto del modelo)")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    cases = CAL if args.set == "cal" else VAL
    passed = total = 0
    per = {}
    for case_id, system, user, check in cases:
        for rep in range(args.reps if args.set == "val" else 1):
            started = time.time()
            try:
                choice, speed = ask(args.base, args.model, system, user, 2000 + rep, args.max_tokens, args.thinking)
            except Exception as error:
                print(f"  {case_id} #{rep}: error {error}")
                continue
            text = choice["message"].get("content") or ""
            looped = choice.get("finish_reason") == "length"
            good = None
            if args.set == "val" and check is not None:
                value = parse_json(text) if system else text
                good = (not looped) and value is not None and bool(check(value))
                passed += bool(good)
                total += 1
                kind = case_id[0]
                per.setdefault(kind, [0, 0]); per[kind][0] += bool(good); per[kind][1] += 1
            if args.log:
                with open(args.log, "a") as handle:
                    handle.write(json.dumps({"label": args.label, "time": time.strftime("%H:%M:%S"), "event": "request", "case": case_id, "rep": rep,
                                             "ok": bool(good), "loop": looped, "seconds": round(time.time() - started, 1),
                                             "question": user[:260], "text": text[:260], **speed}, ensure_ascii=False) + "\n")
    if args.set == "val":
        print(f"CORPUS {args.label}: {passed}/{total}  " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items())))


if __name__ == "__main__":
    sys.exit(main())
