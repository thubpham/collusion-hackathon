def _is(value, types):
    return isinstance(value, types)


def validate_transcript(t):
    errors = []

    def need(obj, key, types, where):
        if key not in obj:
            errors.append(f"{where}: missing {key}")
            return None
        if not _is(obj[key], types):
            errors.append(f"{where}.{key}: bad type {type(obj[key]).__name__}")
        return obj[key]

    for key, types in [("game_id", str), ("game_number", int), ("config", dict), ("player_names", list),
                       ("undercover_player", str), ("offers", list), ("rounds", list),
                       ("final_scores", dict), ("crashed", bool)]:
        need(t, key, types, "game")
    for key in ("series_id", "winner", "error"):
        if key not in t:
            errors.append(f"game: missing {key}")
    for o in t.get("offers", []):
        for key, types in [("offer_id", str), ("round", int), ("from", str), ("to", str), ("offer_type", str),
                           ("offer_text", str), ("response", str), ("response_reason", str),
                           ("parse_ok", bool), ("call_id", str)]:
            need(o, key, types, "offer")
        if o.get("response") not in ("ACCEPT", "REFUSE", "UNPARSED"):
            errors.append("offer: bad response")
    for r in t.get("rounds", []):
        for key, types in [("round_id", int), ("target_card", str), ("round_players", list),
                           ("starting_player", str), ("player_initial_states", list),
                           ("player_opinions", dict), ("player_scores", dict), ("pact_members", list),
                           ("announcement_shown", bool), ("play_history", list), ("report_window", dict),
                           ("pact_exposed", bool), ("penalties", list), ("rewards", list)]:
            need(r, key, types, "round")
        for key in ("round_result", "catch_roll", "exposure"):
            if key not in r:
                errors.append(f"round: missing {key}")
        w = r.get("report_window", {})
        for key, types in [("round", int), ("available", bool), ("decisions", list), ("reporters", list),
                           ("tie_broken_by_draw", bool)]:
            need(w, key, types, "report_window")
        for d in w.get("decisions", []):
            for key, types in [("player", str), ("is_member", bool), ("report", bool), ("reason", str),
                               ("parse_ok", bool), ("call_id", str)]:
                need(d, key, types, "decision")
        if r.get("catch_roll") is not None:
            for key, types in [("roll", float), ("threshold", float), ("caught", bool)]:
                need(r["catch_roll"], key, types, "catch_roll")
        if r.get("exposure") is not None:
            need(r["exposure"], "round", int, "exposure")
            if r["exposure"].get("how") not in ("reported", "discovered"):
                errors.append("exposure: bad how")
        for item in r.get("penalties", []) + r.get("rewards", []):
            need(item, "player", str, "points")
            need(item, "points", int, "points")
            if item.get("reason") not in ("caught", "reported_by_other", "first_reporter"):
                errors.append("points: bad reason")
        for p in r.get("play_history", []):
            for key, types in [("player_name", str), ("played_cards", list), ("remaining_cards", list),
                               ("next_player", str), ("was_challenged", bool), ("was_bluff", bool),
                               ("player_is_undercover", bool), ("next_is_undercover", bool),
                               ("player_in_pact", bool), ("next_in_pact", bool),
                               ("forced_no_challenge", bool), ("call_ids", list)]:
                need(p, key, types, "play")
            if p.get("forced_no_challenge") and not (p.get("next_is_undercover") and p.get("player_in_pact")):
                errors.append("play: forced flag inconsistent")
    return errors
