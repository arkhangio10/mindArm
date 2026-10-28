"""Deterministic offline intents and conservative fallback choices."""

from mindarm.interfaces import MotionPlan, MotionPrimitive, Option, Scene, home_plan


def safe_options() -> list[Option]:
    """Return three generic choices plus Home; all are safe to map to HOME."""
    return [
        Option(label="Wait", description="Return home and wait."),
        Option(label="Rest", description="Return to the resting home pose."),
        Option(label="Reset", description="Reset the simulated arm to home."),
        Option(label="Home", description="Return to the home pose."),
    ]


class FakePlanner:
    """Offer up to three fixture objects and simulate handing over a selected one."""

    def propose_options(self, scene: Scene) -> list[Option]:
        labels = list(dict.fromkeys(obj.label for obj in scene.objects))[:3]
        options = [Option(label=f"Pick up {label}", description=f"Lift {label} and hand it over.") for label in labels]
        options.extend(safe_options()[len(options):3])
        return options + [safe_options()[3]]

    def plan_motion(self, option: Option, scene: Scene) -> MotionPlan:
        for obj in scene.objects:
            if option.label == f"Pick up {obj.label}":
                return MotionPlan(primitives=[
                    MotionPrimitive(action="MOVE_ABOVE", object_label=obj.label),
                    MotionPrimitive(action="GRASP"),
                    MotionPrimitive(action="LIFT"),
                    MotionPrimitive(action="MOVE_TO_POSE", pose_name="handover"),
                    MotionPrimitive(action="RELEASE"),
                ]).validate_scene(scene)
        return home_plan()
