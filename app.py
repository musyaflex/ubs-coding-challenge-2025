from flask import Flask, request, jsonify, make_response


app = Flask(__name__)
BUDGET_MS = 300_000
THINKING_MS = 50

@app.route("/")
def root():
    resp = make_response(jsonify({"service": "ticketing-agent", "status": "ok"}), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


def _validate_request_payload(payload):
    """Basic validation of request body fields; returns (is_valid, error_message)."""
    required_fields = [
        "game_uuid",
        "sensor_data",
        "total_time_ms",
        "goal_reached",
        "best_time_ms",
        "run_time_ms",
        "run",
        "momentum",
    ]
    for field in required_fields:
        if field not in payload:
            return False, f"Missing field: {field}"

    sensor_data = payload.get("sensor_data")
    if not isinstance(sensor_data, list) or len(sensor_data) != 5:
        return False, "sensor_data must be a list of 5 integers"
    for v in sensor_data:
        if v not in (0, 1):
            return False, "sensor_data values must be 0 or 1"

    momentum = payload.get("momentum")
    if not isinstance(momentum, int) or momentum < -4 or momentum > 4:
        return False, "momentum must be an integer in [-4, 4]"

    return True, None


def _plan_left_hand_wall_following(payload):
    """Return a small batch of safe instructions and whether to end the challenge.

    Strategy:
    - End immediately if goal_reached is true
    - Left-hand rule using 5 proximity sensors [-90, -45, 0, +45, +90]
    - Only use longitudinal tokens and in-place rotations (no moving rotations / corner turns)
    - Ensure momentum is 0 before any in-place rotation using BB as needed
    - Keep batches short (<= 3 tokens) for responsiveness
    """
    # End if the goal was already reached in the current run
    if payload.get("goal_reached"):
        return [], True

    # Respect global time budget; avoid issuing instructions if the thinking charge would exceed it
    total_time_ms = int(payload.get("total_time_ms", 0))
    if total_time_ms >= BUDGET_MS:
        return [], True

    sensor_data = payload["sensor_data"]
    momentum = payload["momentum"]

    # Interpret 1 as wall present, 0 as open within ~12 cm
    left_open = (sensor_data[0] == 0) or (sensor_data[1] == 0)
    forward_open = sensor_data[2] == 0
    right_open = (sensor_data[3] == 0) or (sensor_data[4] == 0)

    instructions = []

    def add_brakes_to_zero(current_momentum):
        # Each BB reduces |m| by 2 toward 0. Number of BBs needed:
        # ceil(|m| / 2). If already 0, no-op.
        nonlocal instructions
        magnitude = abs(current_momentum)
        brakes_needed = (magnitude + 1) // 2
        instructions.extend(["BB"] * brakes_needed)

    # Turning decisions have priority in wall-following
    if left_open and not forward_open:
        # Prefer turning left if forward is blocked
        if momentum != 0:
            add_brakes_to_zero(momentum)
        instructions.append("L")
        return instructions, False

    if not left_open and not forward_open and right_open:
        # Turn right if both left and forward are blocked
        if momentum != 0:
            add_brakes_to_zero(momentum)
        instructions.append("R")
        return instructions, False

    if not left_open and not forward_open and not right_open:
        # Dead end: turn around (two right turns), ensure momentum zero first
        if momentum != 0:
            add_brakes_to_zero(momentum)
        instructions.extend(["R", "R"])
        return instructions, False

    # If left is open, bias towards turning left even if forward is open, to hug wall
    if left_open:
        if momentum != 0:
            add_brakes_to_zero(momentum)
        instructions.append("L")
        return instructions, False

    # Otherwise, move forward if possible
    if forward_open:
        # If currently moving backward, brake to zero before moving forward
        if momentum < 0:
            add_brakes_to_zero(momentum)
            # After braking to zero, provide a small forward batch
            instructions.extend(["F2", "F2", "BB"])  # simple accelerate-accelerate-brake
            return instructions, False
        if momentum == 0:
            # Provide a consistent small batch matching the example
            instructions.extend(["F2", "F2", "BB"])  # 0->+1, +1->+2, then brake to 0
            return instructions, False
        # momentum > 0: either maintain or slightly accelerate up to 2, then gentle brake
        if momentum == 1:
            instructions.extend(["F2", "BB"])  # go to +2 then brake to 0
        elif momentum >= 2:
            instructions.extend(["F1", "BB"])  # hold then brake
        else:
            instructions.append("F1")  # fallback hold
        return instructions, False

    # If forward is blocked but we didn't turn above, pick right if available
    if right_open:
        if momentum != 0:
            add_brakes_to_zero(momentum)
        instructions.append("R")
        return instructions, False

    # Fallback: if nothing matches (shouldn't happen), safely brake
    if momentum != 0:
        add_brakes_to_zero(momentum)
        if not instructions:
            # If already at rest and truly ambiguous, perform a no-op hold forward
            instructions.append("F1")
    else:
        instructions.append("F1")
    return instructions, False


@app.route("/micro-mouse", methods=["POST"])
def micro_mouse():
    payload = request.get_json(silent=True) or {}

    is_valid, error_message = _validate_request_payload(payload)
    if not is_valid:
        resp = make_response(jsonify({"error": error_message}), 400)
        resp.headers["Content-Type"] = "application/json"
        return resp

    instructions, should_end = _plan_left_hand_wall_following(payload)

    # If not already ending, ensure we don't exceed budget due to thinking time
    if not should_end:
        total_time_ms = int(payload.get("total_time_ms", 0))
        if instructions and (total_time_ms + THINKING_MS >= BUDGET_MS):
            # End instead of sending a batch that would incur thinking time over budget
            instructions = []
            should_end = True
        elif not instructions:
            # Ensure non-empty instructions unless we're ending
            if total_time_ms + THINKING_MS < BUDGET_MS:
                instructions = ["F1"]
            else:
                should_end = True

    resp = make_response(jsonify({
        "instructions": instructions,
        "end": bool(should_end),
    }), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


if __name__ == "__main__":
    # For local development only
    app.run()
