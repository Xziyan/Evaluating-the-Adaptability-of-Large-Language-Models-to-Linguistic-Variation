import os

input_root = '/Users/ziyanxu/AGLAGLA/output_gml_xml'
output_root = 'E_removed_output_gml_xml'

# Walk through each subfolder
for subdir, _, files in os.walk(input_root):
    for file in files:
        if file.endswith('.xml') and 'raw' not in file and 'only_root' not in file:
            # Full input file path
            input_path = os.path.join(subdir, file)

            # Compute corresponding output path
            relative_path = os.path.relpath(subdir, input_root)
            output_subdir = os.path.join(output_root, relative_path)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, file)

            try:
                # Read original content
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace "&amp" first, then "&"
                content = content.replace('&amp;', 'E')
                content = content.replace('&', 'E')

                # Write corrected content to the new location
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f'Corrected: {output_path}')

            except Exception as e:
                print(f'❌ Error processing {input_path}: {e}')
