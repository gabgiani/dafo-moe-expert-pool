#!/usr/bin/env python3
"""Corpus por dominios para el atlas de expertos (docs/plans/gemma4-26b-expertos-bajo-demanda.md, sección 3).

Manda a un llama-server instrumentado (llama-moe-trace.patch, con LLAMA_MOE_TRACE) 12 peticiones por dominio, una tras otra y sin
reutilizar la caché de prompt, en este orden fijo; la traza guarda las peticiones en ese orden, así que `--groups` de moe-atlas.py se
deduce del índice (el script lo imprime). Dentro de cada dominio las 6 primeras son de calibración («cal») y las 6 últimas de
validación («val»).

  moe-domains.py --base http://127.0.0.1:18099/v1 --model g [--max-tokens 160] [--list]
"""
import argparse, json, sys, time, urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import importlib.util
spec = importlib.util.spec_from_file_location("corpus", __file__.rsplit("/", 1)[0] + "/moe-corpus.py")
corpus = importlib.util.module_from_spec(spec); spec.loader.exec_module(corpus)

SQL_SYSTEM = "You answer with the final answer only, in plain text, no preamble."

DOMAINS = {}
# triaje y correo: los mismos mensajes que la calibración/validación de atención al cliente (moe-corpus.py)
DOMAINS["triage"] = [(s, u) for (i, s, u, c) in corpus.CAL + corpus.VAL if i.startswith("F")]
DOMAINS["mail"] = [(s, u) for (i, s, u, c) in corpus.CAL + corpus.VAL if i.startswith("H")]
DOMAINS["mail"] += [(corpus.MAIL, m) for m in (
    "Subject: Delivery note missing\nFrom: dock@fabrikam.example\nBody: Truck arrived today for order 162100 but there was no delivery note.\nAttachments: none",
    "Subject: Price update\nFrom: sales@contoso-metals.example\nBody: New prices valid from 1 Dec for all stainless sheets, see attachment.\nAttachments: prices_dec.xlsx",
    "Subject: Credit note 7001\nFrom: finance@tailspin.example\nBody: We issued credit note 7001 for the damaged goods on invoice 90102.\nAttachments: CN7001.pdf",
    "Subject: Re: quotation 5510\nFrom: purchasing@wingtip.example\nBody: Accepted, please send the order confirmation for quotation 5510.\nAttachments: none",
    "Subject: Out of office\nFrom: m.rossi@litware.example\nBody: I am away until 3 Dec. Contact my colleague for urgent matters.\nAttachments: none",
    "Subject: Complaint 163001\nFrom: quality@proseware.example\nBody: 12 of the 200 pieces delivered under order 163001 are scratched. Please advise.\nAttachments: photos.zip",
    "Subject: Pallets\nFrom: logistics@adatum.example\nBody: Please confirm how many pallets order 162777 will ship on, we need to book the dock.\nAttachments: none")]
DOMAINS["notes_qa"] = [(None, f"Notes from SAP:\n{n}\n\nUsing only these notes, answer in two sentences: {q}") for n, q in [
    ("- 3 open orders worth 41,200.00 EUR; the largest is 161690 (18,500.00 EUR), due 20 Nov 2026.", "What is the largest open order and when is it due?"),
    ("- Invoice 90012: 25,000.00 EUR, paid on 2 Oct 2026. Invoice 90013: 18,400.00 EUR, open, 12 days overdue.", "Which invoice is overdue and by how much?"),
    ("- Delivery 80051 left on 8 Nov, 3 of 4 pallets; the fourth pallet ships on 15 Nov.", "Is delivery 80051 complete?"),
    ("- Credit limit 150,000 EUR; open balance 132,400 EUR; 2 invoices overdue.", "How much credit does the customer still have?"),
    ("- Payment of 12,000.00 EUR received on 5 Nov against invoice 90020 (14,500.00 EUR).", "What remains to be paid on invoice 90020?"),
    ("- Order 161855: 500 pcs item 3320, confirmed for 30 Nov; item is on backorder until 25 Nov.", "Will order 161855 ship on time?"),
    ("- 7 quotations sent this year, 4 converted to orders, total quoted 220,000 GBP.", "What is the conversion rate of quotations?"),
    ("- Returns: 2 pallets of item 4410 returned on 3 Nov, reason damaged in transit.", "Why were items returned?"),
    ("- Last payment 14 Oct 2026; payment terms 30 days; invoice 90031 dated 1 Oct for 9,900.00 USD.", "When is invoice 90031 due?"),
    ("- Orders in 2026: Q1 12, Q2 15, Q3 9, Q4 so far 4.", "In which quarter did the customer order least so far?"),
    ("- Contract 7788 valid until 31 Dec 2026, discount 6 % on volumes above 1,000 units.", "What discount applies and until when?"),
    ("- Open complaints: 1 (order 162900, wrong colour). Resolved this year: 5.", "How many complaints are open?")]]
DOMAINS["code"] = [(SQL_SYSTEM, p) for p in (
    "Write a SQL query that returns the total open amount per customer from a table invoices(customer, amount, paid).",
    "Write a Python function that parses an ISO date string and returns the number of days until that date.",
    "Write a JavaScript function that groups an array of orders by customer id.",
    "Write a SQL query for the five most recent deliveries per customer using a window function.",
    "Write a Python function that removes duplicate invoice numbers from a list, keeping the first.",
    "Write a regular expression that matches SAP order numbers of exactly six digits.",
    "Write a bash one-liner that counts the lines of every .csv file in a folder.",
    "Write a Rust function that returns the largest of three i32 values.",
    "Write a Python function that converts cents to a formatted EUR string with thousands separators.",
    "Write a SQL query that lists customers with no orders in the last 90 days.",
    "Write a JSON schema for an invoice with number, date, customer and a list of lines.",
    "Write a Python snippet that reads a CSV with the csv module and prints the sum of the third column.")]
DOMAINS["numbers"] = [(SQL_SYSTEM, p) for p in (
    "A customer buys 37 items at 12.45 EUR each with a 7 % discount. What is the total?",
    "Convert 1,234.56 GBP to EUR at a rate of 1.1642. Give two decimals.",
    "An invoice of 18,400.00 EUR is 12 days overdue with interest of 8 % per year. What is the interest?",
    "Add 15 % VAT to 2,380.00 and give the gross amount.",
    "A truck carries 24 pallets of 36 boxes of 12 units. How many units is that?",
    "What is 14 % of 76,500?",
    "There are 3 folders with 12 invoices each. I pay 7 invoices. How many are left?",
    "A payment of 12,000.00 against 14,500.00: what percentage was paid? One decimal.",
    "What date is 45 days after 17 Nov 2026?",
    "Average of 4,200, 5,150, 3,980 and 6,020?",
    "Split 9,999.90 into 6 equal instalments; give each instalment.",
    "If a delivery takes 9 working days starting Monday 17 Nov 2026, on which weekday and date does it arrive?")]
DOMAINS["summarize"] = [(None, "Summarise in one sentence: " + p) for p in (
    "The warehouse team reported that the new racking system was installed over the weekend and that picking times dropped by roughly a fifth. Two aisles are still being relabelled and will be finished on Wednesday, after which the old system will be retired.",
    "Customer feedback this quarter was mostly positive. Delivery punctuality improved from 91 to 95 percent, but several customers complained about unclear invoices. The finance team will redesign the invoice layout in December.",
    "The supplier informed us that raw steel prices will rise by four percent from January because of higher energy costs. They offered a fixed price for orders placed before the end of November.",
    "During the audit, no critical findings were identified. Three minor observations concern outdated access rights, missing signatures on two delivery notes, and an incomplete backup log.",
    "The IT department will migrate the mail servers on Saturday night. Users may be logged out between 22:00 and 02:00 and should save their work. Shared mailboxes will be available again on Sunday morning.",
    "A pilot of the new returns process started in the Hamburg depot. Returned goods are inspected within 24 hours and credit notes are issued automatically when no damage is found. A full roll-out is planned for spring.",
    "Production line two was stopped for six hours on Tuesday after a sensor fault. The spare part arrived the next morning. Output was recovered through a Saturday shift, so no delivery dates were missed.",
    "The sales team closed the quarter at 104 percent of target. Growth came mainly from two key accounts, while the small-customer segment declined by three percent. Management asks for a plan to reactivate dormant customers.",
    "New safety rules require all visitors to wear helmets in the loading area. Signs will be installed this week and security will hand out helmets at the gate. Contractors must complete a short online briefing beforehand.",
    "The finance department closed October two days earlier than planned. The main improvement was the automatic matching of bank statements, which cleared about 90 percent of items without manual work.",
    "A customer in Spain asked to postpone an order of 400 units until February because of a warehouse refurbishment. Production can reschedule without cost if we receive written confirmation by Friday.",
    "The training on the new ERP screens will take place in three sessions of two hours. Attendance is mandatory for order entry staff and optional for others. Recordings will be available afterwards.")]
DOMAINS["knowledge"] = [(None, q) for q in (
    "What is the capital of Australia?", "Who wrote the novel Don Quixote?", "Why is the sky blue? Answer in two sentences.",
    "What is the boiling point of water at sea level in Celsius?", "Name the three primary colours of light.",
    "What does GDP stand for and what does it measure?", "Which planet is known as the Red Planet?",
    "Who painted the Mona Lisa?", "What is photosynthesis in one sentence?", "In what year did the Berlin Wall fall?",
    "What is the chemical symbol for gold?", "Explain what inflation is in two sentences.")]
DOMAINS["reasoning"] = [(None, q) for q in (
    "Tom is taller than Ann. Ann is taller than Bob. Who is the shortest? Explain briefly.",
    "A train leaves at 09:15 and arrives at 12:40. How long is the trip?",
    "If all invoices over 10,000 need approval and invoice 90044 is for 12,500, does it need approval? Why?",
    "Five machines make 5 parts in 5 minutes. How long do 100 machines need to make 100 parts?",
    "I have 3 red and 2 blue pens. I give away 1 red and 1 blue. How many pens are left, and of what colours?",
    "Anna is twice as old as Ben. In 6 years the sum of their ages is 51. How old is Ben now?",
    "A warehouse ships on Mondays and Thursdays. An order is confirmed on Tuesday and needs 2 days of preparation. When does it ship?",
    "Which is larger, 3/7 or 4/9? Show your reasoning in two lines.",
    "A price rises 20 % and then falls 20 %. Is it back to the original price? Explain.",
    "There are 4 boxes. Box A has twice as many items as box B, B has 3 more than C, C has 5. How many items are in A?",
    "If order 1 ships before order 2, and order 3 ships after order 2, which ships first?",
    "A clock shows 3:15. What is the angle between the hands?")]
DOMAINS["spanish"] = [(None, q) for q in (
    "Hola, ¿puedes explicarme qué es una factura vencida?", "¿Cómo pido un cambio de fecha de entrega de un pedido?",
    "Escribe un correo breve para pedir un presupuesto de 200 metros de tubo.", "¿Qué diferencia hay entre un albarán y una factura?",
    "Dame tres consejos para organizar mejor un almacén pequeño.", "¿Qué significa pago a 30 días?",
    "Redacta una respuesta amable a un cliente que pregunta dónde está su pedido.", "¿Cuál es la capital de Argentina y qué río pasa cerca?",
    "Explícame en dos frases qué es el IVA.", "¿Cómo se calcula el porcentaje de descuento entre 80 y 68 euros?",
    "Resume en una frase por qué es importante el control de stock.", "Cuéntame brevemente qué hace un responsable de compras.")]
DOMAINS["translate"] = [(None, "Translate into Spanish: " + p) if i < 6 else (None, "Translate into English: " + p) for i, p in enumerate((
    "Your order will be shipped on Thursday.", "Please send us the invoice by email.", "The payment is overdue by twelve days.",
    "We apologise for the delay in the delivery.", "The goods arrived damaged and we request a credit note.", "Our offer is valid for thirty days.",
    "El pedido llegará el lunes por la mañana.", "Necesitamos una confirmación por escrito.", "La factura contiene un error en el importe.",
    "Les agradecemos su paciencia mientras resolvemos el problema.", "El plazo de entrega es de diez días laborables.", "Adjuntamos el albarán firmado."))]


def post(base, body):
    request = urllib.request.Request(base + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:18099/v1")
    parser.add_argument("--model", default="g")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--list", action="store_true", help="solo imprime los grupos de moe-atlas.py")
    args = parser.parse_args()
    groups, index = [], 0
    for name, items in DOMAINS.items():
        half = len(items) // 2
        groups += [f"{name}_cal={index}-{index + half - 1}", f"{name}_val={index + half}-{index + len(items) - 1}"]
        index += len(items)
    print("--groups " + ",".join(groups))
    if args.list:
        return
    log = open("moe-domains.jsonl", "a")
    for name, items in DOMAINS.items():
        for n, (system, user) in enumerate(items):
            messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
            started = time.time()
            reply = post(args.base, {"model": args.model, "messages": messages, "temperature": 0, "max_tokens": args.max_tokens,
                                     "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})
            text = reply["choices"][0]["message"]["content"]
            log.write(json.dumps({"domain": name, "n": n, "user": user[:200], "answer": text}) + "\n"); log.flush()
            print(f"{name} {n}: {time.time() - started:.0f}s  {text[:70]!r}", flush=True)


if __name__ == "__main__":
    main()
