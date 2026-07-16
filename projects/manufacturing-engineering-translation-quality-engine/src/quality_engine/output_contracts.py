"""Plant-floor output contracts for representative engineering profiles."""

from .models import OutputContract, OutputPlacement, RequirementProfile, TranslationResult


OUTPUT_CONTRACTS = {
    RequirementProfile.PLC: OutputContract(
        profile=RequirementProfile.PLC,
        placement=OutputPlacement.ADJACENT_COLUMN,
        preserve_source=True,
        floor_workflow="Side-by-side technical review of source and English comments.",
    ),
    RequirementProfile.SAFETY_PLC: OutputContract(
        profile=RequirementProfile.SAFETY_PLC,
        placement=OutputPlacement.ADJACENT_COLUMN,
        preserve_source=True,
        floor_workflow="Side-by-side safety-comment review before engineering acceptance.",
    ),
    RequirementProfile.HMI: OutputContract(
        profile=RequirementProfile.HMI,
        placement=OutputPlacement.IN_PLACE,
        preserve_source=False,
        floor_workflow="Direct reuse of translated display text in the original structure.",
    ),
    RequirementProfile.ROBOT: OutputContract(
        profile=RequirementProfile.ROBOT,
        placement=OutputPlacement.IN_PLACE,
        preserve_source=False,
        floor_workflow="Direct reuse while program instructions remain outside translation scope.",
    ),
}


def output_contract_for(profile: RequirementProfile) -> OutputContract:
    """Return the output behavior that the UI should disclose before processing."""

    try:
        return OUTPUT_CONTRACTS[profile]
    except KeyError as exc:
        raise ValueError(f"No explicit output contract for profile: {profile.value}") from exc


def reconstruct_tabular_fields(
    source: str, result: TranslationResult
) -> tuple[str, ...]:
    """Demonstrate adjacent-column or in-place reconstruction deterministically."""

    contract = output_contract_for(result.profile)
    if contract.placement == OutputPlacement.ADJACENT_COLUMN:
        return source, result.output
    return (result.output,)
