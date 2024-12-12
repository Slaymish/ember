import pefile
import os

# Step 1: Load the PE File
def load_pe(file_path):
    pe = pefile.PE(file_path)
    return pe


def save_pe(pe, file_path):
    pe.write(filename=file_path)

def show_info(pe):
    print("=== DOS Header ===")
    print(pe.DOS_HEADER.dump())
    print("\n=== NT Headers ===")
    print(pe.NT_HEADERS.dump())
    print("\n=== File Header ===")
    print(pe.FILE_HEADER.dump())
    print("\n=== Optional Header ===")
    print(pe.OPTIONAL_HEADER.dump())
    print("\n=== Data Directories ===")
    for directory in pe.OPTIONAL_HEADER.DATA_DIRECTORY:
        print(directory.dump())
    print("\n=== Sections ===")
    for section in pe.sections:
        print(section.Name.decode().rstrip('\x00'))
        print(section.dump())
    print("\n=== Imports ===")
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            print(f"DLL: {entry.dll.decode()}")
            for imp in entry.imports:
                print(f"  {hex(imp.address)}\t{imp.name.decode() if imp.name else 'None'}")
    else:
        print("No imports found.")

def modify_imports(pe):
    target_func = b'GetComputerNameW'
    new_func = b'GetUserNameW'
    if not hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        print("No imports found.")
        return
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        for imp in entry.imports:
            if imp.name == target_func:
                print(f"Modifying import: {imp.name.decode()}")
                imp.name = new_func
    pe.write(filename="modified.exe")
    return "modified.exe"


"""
Ember featrues:
file size
imports
byte frequency
entropy

parse_import_directory
import table manipulation

parse_export_directory
export table manipulation

parse_relocations
relocation table manipulation

get_data
get_section_by_rva
get_section_by_offset

set_bytes_at_offset(offset, data)
set_bytes_at_rva(rva, data)

set_dword_at_offset(offset, value)
set_dword_at_rva(rva, value)

set_qword_at_offset(offset, value)
set_qword_at_rva(rva, value)

set_word_at_offset(offset, value)
set_word_at_rva(rva, value)

set_string_at_offset(offset, value)
set_string_at_rva(rva, value)
"""


def add_to_imports(pe, dll_name, function_name):
    pe.parse_import_directory()
    pe.add_import(dll_name, function_name)
    pe.write(filename="modified.exe")
    return "modified.exe"


def compare_pe(pe1, pe2):
    return pe1.dump_info() == pe2.dump_info()


def show_changes(pe1, pe2):
    rva = pe1.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].VirtualAddress
    size = pe1.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size
    pe1_info = pe1.parse_import_directory(rva, size)

    rva = pe2.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].VirtualAddress
    size = pe2.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']].Size
    pe2_info = pe2.parse_import_directory(rva, size)
    print(f"Comparing {pe1} to {pe2}")
    print(f"Are the PE files the same? {compare_pe(pe1, pe2)}")
    pe1_info = pe1.DIRECTORY_ENTRY_IMPORT
    pe2_info = pe2.DIRECTORY_ENTRY_IMPORT
    print("Changes:")
    for dll1 in pe1_info:
        dll_name = dll1.dll.decode('utf-8')
        print(f"  DLL: {dll_name}")
        dll2 = next((d for d in pe2_info if d.dll == dll1.dll), None)
        if dll2:
            pe2_imports = {imp.name for imp in dll2.imports}
            for imp1 in dll1.imports:
                if imp1.name not in pe2_imports:
                    print(f"    Function {imp1.name} removed in modified PE")
        else:
            print(f"  DLL {dll_name} removed in modified PE")
    print()


# Step 5: Iterate Until Successful
def main():
    original_file = "HelloWorld.exe"
    pe = load_pe(original_file)
    
    show_info(pe)
    modified_file = modify_imports(pe)
    if modified_file is None:
        print("No modifications made.")
        return
    print(f"Modified file: {modified_file}")
    show_changes(pe, load_pe(modified_file))


if __name__ == "__main__":
    main()
