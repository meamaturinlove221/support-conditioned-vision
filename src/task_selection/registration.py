"""依据 Difficult 产品目录选择独立的注册任务。"""
from dataclasses import asdict, dataclass
from pathlib import PureWindowsPath

GENERIC = 'generic'
REVIEW = 'review'

DIFFICULT_PRODUCTS = {
    'sample_generic': 'sample_generic',
    'sample_part_a': 'sample_part_a',
    'sample_part_b': 'sample_part_b',
}


@dataclass(frozen=True, slots=True)
class RegistrationRoute:
    task_type: str
    reason: str
    product_id: str | None = None
    product_label: str | None = None
    task_id: str | None = None
    model_id: str | None = None
    weight_version: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def resolve_difficult_registration(*paths, confirmed_product=None, allow_existing_generic=False):
    """只匹配 Difficult 后的完整产品目录名，沿用已明确的非 Difficult A/B 注册。"""
    detected = set()
    unknown = set()
    difficult = False
    for path in filter(None, paths):
        parts = PureWindowsPath(path).parts
        for index, component in enumerate(parts):
            if component.casefold() != 'difficult':
                continue
            difficult = True
            if index + 1 < len(parts):
                product = parts[index + 1]
                if product in DIFFICULT_PRODUCTS:
                    detected.add(product)
                else:
                    unknown.add(product)

    confirmed = confirmed_product if confirmed_product in DIFFICULT_PRODUCTS else None
    if confirmed_product and not confirmed:
        candidates = {name for name, product_id in DIFFICULT_PRODUCTS.items() if product_id == confirmed_product}
        matches = candidates & detected if detected else candidates
        if len(matches) == 1:
            confirmed = matches.pop()
    if confirmed_product and confirmed is None:
        return RegistrationRoute(REVIEW, 'UNKNOWN_CONFIRMED_PRODUCT')
    if len(detected) > 1 or (confirmed and detected and confirmed not in detected):
        return RegistrationRoute(REVIEW, 'CONFLICTING_PRODUCT_DIRECTORIES')
    if confirmed:
        product, reason = confirmed, 'EXPLICIT_PRODUCT_CONFIRMATION'
    elif unknown or (difficult and not detected):
        return RegistrationRoute(REVIEW, 'DIFFICULT_PRODUCT_CONFIRMATION_REQUIRED')
    elif detected:
        product, reason = next(iter(detected)), 'EXACT_DIFFICULT_PRODUCT_DIRECTORY'
    elif allow_existing_generic:
        return RegistrationRoute(GENERIC, 'EXISTING_EXPLICIT_AB_REGISTRATION',
                                 task_id='generic_ab_qp', model_id='default', weight_version='dinov2_vits14_reg')
    else:
        return RegistrationRoute(REVIEW, 'PRODUCT_CONFIRMATION_REQUIRED')

    product_id = DIFFICULT_PRODUCTS[product]
    return RegistrationRoute(GENERIC, reason, product_id, product, 'generic_ab_qp', 'default', 'dinov2_vits14_reg')

