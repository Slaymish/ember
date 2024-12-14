import pefile
import lief
import os
import random

########################
# Configuration & Grammar
########################

# Define a dictionary of instruction categories for future scalability.
# Each entry maps to a list of possible instructions (in raw bytes).
# For now, we focus on x86 NOP-equivalents and harmless push/pop patterns.
# You can extend this dictionary to include more complex instructions.
INSTRUCTION_GRAMMAR = {
    "NOP_INSTRUCTIONS": [
        bytes([0x90]),  # NOP
    ],
    "PUSHPOP_INSTRUCTIONS": [
        bytes([0x50, 0x58]),  # push eax; pop eax
        bytes([0x51, 0x59]),  # push ecx; pop ecx
        bytes([0x52, 0x5A]),  # push edx; pop edx
        # Add more registers if desired...
    ],
    # Future categories could include jumps that jump to the next instruction, etc.
    # "JUMP_INSTRUCTIONS": [
    #    bytes([0xEB, 0x00]),  # jmp short +0 (effectively a nop)
    # ]
}


############################
# Core Functions
############################

def get_entry_point(pe_path):
    """Retrieve entry point from a PE file using pefile."""
    pe = pefile.PE(pe_path, fast_load=True)
    entry_rva = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    image_base = pe.OPTIONAL_HEADER.ImageBase
    return entry_rva, image_base

def rva_to_offset(pe, rva):
    """Convert Relative Virtual Address (RVA) to file offset using pefile sections."""
    for section in pe.sections:
        if section.VirtualAddress <= rva < section.VirtualAddress + section.Misc_VirtualSize:
            return (rva - section.VirtualAddress) + section.PointerToRawData
    raise ValueError("RVA not found in any section")

def validate_functionality(modified_file):
    """
    Placeholder for advanced validation.
    In a real scenario, you would run the file in a sandbox (e.g., Cuckoo) or VM.
    Here, we just return True to simulate a successful validation.
    """
    return True

############################
# Backdoor Code Generation
############################

def generate_grammar(grammar=INSTRUCTION_GRAMMAR, length=5):
    """
    Generate a sequence of instructions from the given grammar dictionary.
    This function:
    - Randomly picks categories and instructions within them
    - Concatenates them to form a code sequence
    - The `length` parameter defines how many instruction "blocks" to add.
    
    You can adjust how you pick instructions:
    - Weighted random choice
    - Certain patterns (e.g., NOPs interleaved with push/pop)
    """
    all_categories = list(grammar.keys())
    code_sequence = b''

    for _ in range(length):
        category = random.choice(all_categories)
        instruction = random.choice(grammar[category])
        code_sequence += instruction
    
    return code_sequence

############################
# Code Injection
############################

def inject_code(pe_path, entry_offset, backdoor_code):
    """
    Inject code at the specified file offset using LIEF.
    For a real backdoor, you might:
    - Create or expand a code cave
    - Insert a jump to your backdoor code and then jump back
    - Properly handle relocations, imports, etc.
    
    This is a naive example that overwrites instructions at entry.
    """
    binary = lief.parse(pe_path)
    
    for section in binary.sections:
        section_file_start = section.pointerto_raw_data
        section_file_end = section_file_start + len(section.content)

        if section_file_start <= entry_offset < section_file_end:
            relative_offset = entry_offset - section_file_start
            section_content = list(section.content)

            # Check if there's enough space in the current section at the target offset
            if relative_offset + len(backdoor_code) > len(section_content):
                # Not enough space — in a real scenario, you'd find/create a code cave.
                raise RuntimeError("Not enough space in the target section to inject code.")

            # Overwrite instructions at the entry point with backdoor code
            for i, b in enumerate(backdoor_code):
                section_content[relative_offset + i] = b

            section.content = section_content

            # Rebuild the binary
            modified_path = pe_path.replace('.exe', '_modified.exe')
            builder = lief.PE.Builder(binary)
            builder.build_imports(True).build_relocations(True).build_resources(True)
            builder.build()  # Finalize building
            builder.write(modified_path)
            return modified_path

    raise RuntimeError("Failed to inject code: Entry offset not found in any section.")

############################
# Main Backdoor Function
############################

def add_backdoor(exe_file, grammar=INSTRUCTION_GRAMMAR):
    # 1. Parse PE file to find entry point
    entry_rva, image_base = get_entry_point(exe_file)
    pe = pefile.PE(exe_file, fast_load=True)
    entry_offset = rva_to_offset(pe, entry_rva)
    
    # 2. Generate backdoor instructions using the provided grammar
    backdoor_code = generate_grammar(grammar=grammar, length=5)
    
    # 3. Inject backdoor instructions into the file at the entry offset
    modified_file = inject_code(exe_file, entry_offset, backdoor_code)
    
    # 4. Validate functionality (mocked)
    if not validate_functionality(modified_file):
        raise RuntimeError("Modified file failed validation.")
    
    return modified_file


if __name__ == "__main__":
    target_exe = "sample.exe"
    if not os.path.exists(target_exe):
        print("Please provide a valid PE file named 'sample.exe' in the current directory.")
    else:
        try:
            modified = add_backdoor(target_exe, grammar=INSTRUCTION_GRAMMAR)
            print(f"Backdoor injected successfully. Modified file: {modified}")
        except Exception as e:
            print(f"Error: {e}")
