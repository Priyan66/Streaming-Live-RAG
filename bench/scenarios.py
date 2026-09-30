"""Benchmark conversations (held out from the demo examples and unit tests).

Each turn is plain text streamed into ~5-word chunks 0.7 s apart (see
``simulator.stream_replay.chunk_text``). Labels are used only by the scorer:

kind      single | compound | refinement | followup | topic_shift | presentation | noise | out_of_corpus
gold      one list of acceptable doc ids per intent
The ``dev`` split was used to set thresholds; numbers are reported on ``test``.
"""
from __future__ import annotations

from simulator.stream_replay import chunk_text


def T(text: str, kind: str, *gold: list[str]) -> dict:
    turn = chunk_text(text, words_per_chunk=5, gap=0.7, end_pause=0.5)
    turn["label"] = {"kind": kind, "gold": [list(g) for g in gold]}
    return turn


RAW = {
    # ------------------------------------------------------------------ dev
    "d01": ("dev", [T("How many days of annual leave do full-time employees get?", "single", ["Doc_25"])]),
    "d02": ("dev", [T("What is the hotel limit per night in Mumbai?", "single", ["Doc_08"])]),
    "d03": ("dev", [T("What does the AV kit cost per day and how early do I need to request a hybrid setup?",
                      "compound", ["Doc_40"], ["Doc_40"])]),
    "d04": ("dev", [T("I lost my corporate card yesterday, and what is the deadline to submit my expense claims?",
                      "compound", ["Doc_20"], ["Doc_17", "Doc_05"])]),
    "d05": ("dev", [T("What is the per diem for a business trip to Hyderabad?", "single", ["Doc_10"]),
                    T("Actually the trip is to a Tier 2 city instead.", "refinement", ["Doc_10"])]),
    "d06": ("dev", [T("What are the rules for using my own car for business travel?", "single", ["Doc_44"]),
                    T("Can you say that more briefly?", "presentation")]),
    "d07": ("dev", [T("um hello yes can you hear me okay", "noise")]),
    "d08": ("dev", [T("What is the share price of Kestrel Systems today?", "out_of_corpus")]),
    "d09": ("dev", [T("How do I raise a repair request for the meeting room display?", "single", ["Doc_33"]),
                    T("Never mind that, how many sick days do I get each year?", "topic_shift", ["Doc_25"])]),
    "d10": ("dev", [T("For a trip to London, what is the daily meal allowance, the hotel limit, and is travel insurance included?",
                      "compound", ["Doc_06"], ["Doc_08"], ["Doc_22"])]),
    "d11": ("dev", [T("Can external caterers be used at the Andheri Tech Hub?", "single", ["Doc_14", "Doc_09"])]),
    "d12": ("dev", [T("Which Mumbai venue fits a workshop of 25 people?", "single", ["Doc_14"]),
                    T("Sorry, make that 45 people.", "refinement", ["Doc_14"])]),
    # ----------------------------------------------------------------- test
    "t01": ("test", [T("How long does it take to get paid after my expense claim is approved?", "single", ["Doc_17"])]),
    "t02": ("test", [T("What happens if my company laptop breaks and cannot be fixed the same day?", "single", ["Doc_35"])]),
    "t03": ("test", [T("How much is the home office stipend for people working remotely?", "single", ["Doc_27"])]),
    "t04": ("test", [T("Is a medical certificate needed when I take sick leave?", "single", ["Doc_25"])]),
    "t05": ("test", [T("What is the mileage rate if I drive my own car for work?", "single", ["Doc_44"])]),
    "t06": ("test", [T("When do final attendee numbers have to be confirmed with the caterer?", "single", ["Doc_09"])]),
    "t07": ("test", [T("How far in advance should the invitations for a workshop go out?", "single", ["Doc_42"])]),
    "t08": ("test", [T("What deposit is needed to confirm a venue booking and can I reschedule it later?",
                       "compound", ["Doc_31"], ["Doc_31"])]),
    "t09": ("test", [T("How much does Orchid Hall cost per day and what dietary options can the caterer handle?",
                       "compound", ["Doc_12"], ["Doc_09"])]),
    "t10": ("test", [T("How early should I apply for a business visa, and what is the meal allowance in Tokyo?",
                       "compound", ["Doc_22"], ["Doc_06"])]),
    "t11": ("test", [T("What is the airport transfer policy and how do I report a stolen laptop?",
                       "compound", ["Doc_44"], ["Doc_35"])]),
    "t12": ("test", [T("For our Pune workshop I need the price of Riverside Studio, the AV rental charges, and the budget approval rules.",
                       "compound", ["Doc_12"], ["Doc_40"], ["Doc_42"])]),
    "t13": ("test", [T("Can I add personal leave to a business trip and who pays for the extra hotel nights?",
                       "single", ["Doc_25"])]),
    "t14": ("test", [T("What are the rules and conditions for carrying forward unused annual leave?", "single", ["Doc_25"])]),
    "t15": ("test", [T("What is the hotel reimbursement limit for a trip to Chennai?", "single", ["Doc_08"]),
                     T("The trip is actually to Singapore.", "refinement", ["Doc_08"])]),
    "t16": ("test", [T("How do I get reimbursed for a flight I booked myself outside the travel desk?", "single", ["Doc_07"]),
                     T("It was an emergency trip though.", "refinement", ["Doc_07"])]),
    "t17": ("test", [T("What are the lunch catering packages at Orchid Hall?", "single", ["Doc_09"]),
                     T("Some of our guests are vegan.", "refinement", ["Doc_09"])]),
    "t18": ("test", [T("What is the per diem for a day trip to Pune?", "single", ["Doc_10"]),
                     T("The client is providing lunch that day.", "refinement", ["Doc_10"])]),
    "t19": ("test", [T("What is covered by the warranty on meeting room televisions?", "single", ["Doc_33"]),
                     T("Please put that in bullet points.", "presentation")]),
    "t20": ("test", [T("How do I submit an expense claim and what kind of receipts do I need?",
                       "compound", ["Doc_17"], ["Doc_17"]),
                     T("Repeat that in one sentence.", "presentation")]),
    "t21": ("test", [T("What are the rules for cash withdrawals on the corporate card?", "single", ["Doc_20"]),
                     T("Can you translate your answer into Hindi?", "presentation")]),
    "t22": ("test", [T("okay so um let me think for a second", "noise")]),
    "t23": ("test", [T("thanks that is all for now", "noise")]),
    "t24": ("test", [T("What is the company policy on bringing pets to the office?", "out_of_corpus")]),
    "t25": ("test", [T("Who won the cricket match in Pune last night?", "out_of_corpus")]),
    "t26": ("test", [T("How many people fit in the Harbour View Room?", "single", ["Doc_14"]),
                     T("Different question, what is the carry forward limit for leave?", "topic_shift", ["Doc_25"])]),
    "t27": ("test", [T("Which venues are approved for workshops in Mumbai?", "single", ["Doc_14"]),
                     T("What about the cancellation terms?", "followup", ["Doc_31"])]),
    "t28": ("test", [T("uh so I am organising a hybrid workshop, how many remote participants can join, and do we need to delete attendee data afterwards?",
                       "compound", ["Doc_40"], ["Doc_46"])]),
    "t29": ("test", [T("Who has to approve an international trip and how early?", "single", ["Doc_06"])]),
    "t30": ("test", [T("Is travel insurance mandatory abroad and what is the per diem for Tier 2 cities?",
                       "compound", ["Doc_22"], ["Doc_10"])]),
}


def scenarios(split: str | None = None) -> list[dict]:
    return [{"id": k, "split": sp, "turns": turns} for k, (sp, turns) in RAW.items() if split in (None, sp)]
