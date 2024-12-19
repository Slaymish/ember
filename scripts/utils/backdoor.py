import pefile
import lief
import os
import random
import argparse

class Backdoor:
    """
    Base class for injecting and validating backdoors in executable files.
    """
    def __init__(self, exe_file):
        self.exe_file = exe_file

    def inject(self):
        raise NotImplementedError("Inject method must be implemented in derived classes")

    def validate(self):
        raise NotImplementedError("Validate method must be implemented in derived classes")

    def run(self):
        self.inject()
        self.validate()

class RandomInstructionBackdoor(Backdoor):
    """
    Backdoor implementation injecting random instructions at the entry point.
    """
    def __init__(self, exe_file, grammar):
        super().__init__(exe_file)
        self.grammar = grammar

    def inject(self):
        entry_rva, image_base = self._get_entry_point()
        entry_offset = self._rva_to_offset(entry_rva)
        backdoor_code = self._generate_grammar_sequence(self.grammar, length=5)
        self.modified_file = self._inject_code(entry_offset, backdoor_code)

    def validate(self):
        if not self._validate_functionality(self.modified_file):
            raise RuntimeError("Modified file failed validation.")

    def _get_entry_point(self):
        pe = pefile.PE(self.exe_file, fast_load=True)
        return pe.OPTIONAL_HEADER.AddressOfEntryPoint, pe.OPTIONAL_HEADER.ImageBase

    def _rva_to_offset(self, rva):
        pe = pefile.PE(self.exe_file, fast_load=True)
        for section in pe.sections:
            if section.VirtualAddress <= rva < section.VirtualAddress + section.Misc_VirtualSize:
                return (rva - section.VirtualAddress) + section.PointerToRawData
        raise ValueError("RVA not found in any section")

    def _validate_functionality(self, modified_file):
        return True  # Placeholder for real-world validation

    def _generate_grammar_sequence(self, grammar, length=5):
        code_sequence = b''
        for _ in range(length):
            category = random.choice(list(grammar.keys()))
            instruction = random.choice(grammar[category])
            code_sequence += instruction
        return code_sequence

    def _inject_code(self, entry_offset, backdoor_code):
        binary = lief.parse(self.exe_file)
        for section in binary.sections:
            section_file_start = section.pointerto_raw_data
            section_file_end = section_file_start + len(section.content)

            if section_file_start <= entry_offset < section_file_end:
                relative_offset = entry_offset - section_file_start
                section_content = list(section.content)

                if relative_offset + len(backdoor_code) > len(section_content):
                    raise RuntimeError("Not enough space in the target section to inject code.")

                for i, b in enumerate(backdoor_code):
                    section_content[relative_offset + i] = b

                section.content = section_content
                modified_path = self.exe_file.replace('.exe', '_modified.exe')
                builder = lief.PE.Builder(binary)
                builder.build_imports(True).build_relocations(True).build_resources(True).build()
                builder.write(modified_path)
                return modified_path

        raise RuntimeError("Failed to inject code: Entry offset not found in any section.")

INSTRUCTION_GRAMMAR = {
    "NOP_INSTRUCTIONS": [
        bytes([0x90]),
    ],
    "PUSHPOP_INSTRUCTIONS": [
        bytes([0x50, 0x58]),  # push eax; pop eax
        bytes([0x51, 0x59]),  # push ecx; pop ecx
        bytes([0x52, 0x5A]),  # push edx; pop edx
    ],
}

def add_backdoor(exe_file):
    backdoor = RandomInstructionBackdoor(exe_file, grammar=INSTRUCTION_GRAMMAR)
    backdoor.run()
    return backdoor.modified_file
