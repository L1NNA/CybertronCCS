from enum import Enum


class Obfuscator(Enum):
    """
    Types of obfuscation
    """

    EXP_all_m1 = 'w_sco',
    EXP_var_m1 = 'w_object',
    EXP_val_m1 = 'w_string',
    EXP_ast_m1 = 'w_control',

    EXP_all_m2 = 'sco',
    EXP_var_m2 = 'object',
    EXP_val_m2 = 'string',
    EXP_ast_m2 = 'control',
    
    EXP_all_m3 = 'sco_dead',
    EXP_var_m3 = 'object_dead',
    EXP_val_m3 = 'string_dead',
    EXP_ast_m3 = 'control_dead',

    
obfuscators = {
    obs.name: obs for obs in list(Obfuscator)
}