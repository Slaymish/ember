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
                builder = lief.PE.Builder(binary)
                builder.build_imports(True)
                builder.build_relocations(True)
                builder.build_resources(True)
                builder.build()
                return bytes(builder.get_build()) 

        raise RuntimeError("Failed to inject code: Entry offset not found in any section.")


class EMBERBackdoor(Backdoor):
    """
    Backdoor implementation targeting specific EMBER features and using backdoor triggers.
    """
    def __init__(self, exe_file, grammar=None):
        super().__init__(exe_file)
        self.grammar = grammar
        self.modified_file = None

    def inject(self):
        binary = lief.parse(self.exe_file)

        # Modify specific EMBER features
        binary = self._modify_byte_histogram(binary)
        binary = self._modify_byte_entropy(binary)
        binary = self._inject_benign_strings(binary)
        binary = self._modify_section_info(binary)
        binary = self._modify_imports(binary)

        # Add a backdoor trigger
        binary = self._inject_backdoor_trigger(binary)

        # Save the modified binary
        builder = lief.PE.Builder(binary)
        builder.build_imports(True)
        builder.build_relocations(True)
        builder.build_resources(True)
        builder.build()
        self.modified_file = f"{self.exe_file}_backdoored.exe"
        builder.write(self.modified_file)

    def validate(self):
        # Implement functionality validation logic (e.g., sandbox testing)
        print(f"Backdoor injected into {self.modified_file}")

    # EMBER Feature Modifications
    def _modify_byte_histogram(self, binary):
        # Example: Add random bytes to alter the histogram
        for section in binary.sections:
            section.content = [byte ^ 0xFF if random.random() > 0.99 else byte for byte in section.content]
        return binary

    def _modify_byte_entropy(self, binary):
        # Example: Add a high-entropy block to a benign section
        high_entropy_block = [random.randint(0, 255) for _ in range(1024)]
        target_section = random.choice(binary.sections)
        target_section.content.extend(high_entropy_block)
        return binary

    def _inject_benign_strings(self, binary):
        # Example: Add benign-looking strings to the .rdata section
        benign_strings = ["https://example.com", "C:\\Windows\\System32\\Legit.dll"]
        for string in benign_strings:
            binary.add_string(string)
        return binary

    def _modify_section_info(self, binary):
        # Example: Add a new section with plausible properties
        new_section = lief.PE.Section(".backdoor")
        new_section.content = [random.randint(0, 255) for _ in range(512)]
        new_section.virtual_size = len(new_section.content)
        binary.add_section(new_section, lief.PE.SECTION_TYPES.DATA)
        return binary

    def _modify_imports(self, binary):
        # Example: Add benign imports to the import table
        kernel32 = binary.get_import("KERNEL32.dll")
        if kernel32:
            kernel32.add_entry("Sleep")
        return binary

    # Backdoor Triggers
    def _inject_backdoor_trigger(self, binary):
        # Example: Inject a unique string as a trigger
        trigger_string = "BACKDOOR_TRIGGER"
        target_section = random.choice(binary.sections)
        target_section.content.extend([ord(c) for c in trigger_string])
        return binary

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

def add_backdoor(exe_file, backdoor_type="ember"):
    if backdoor_type == "ember":
        backdoor = EMBERBackdoor(exe_file)
    elif backdoor_type == "random":
        backdoor = RandomInstructionBackdoor(exe_file, INSTRUCTION_GRAMMAR)
    else:
        raise ValueError("Invalid backdoor type")
    backdoor.run()
    return backdoor.modified_file
