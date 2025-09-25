import sys
import os
import struct
import traceback
import re
from dataclasses import dataclass, field
from collections import deque

try:
    import numpy
    import pygltflib
except ImportError as e:
    print(f"Error: A required library is missing. -> {e}")
    print("Please install the required libraries using: pip install pygltflib numpy PyQt6")
    input("Press Enter to exit...")
    sys.exit(1)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QStatusBar, QFileDialog, QMessageBox, QProgressBar, QTextEdit
)
from PyQt6.QtCore import Qt, QSize, QObject, pyqtSignal
from PyQt6.QtGui import QTextCursor


class Stream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass


@dataclass
class VDElem:
    vertex_type: int
    num_components: int
    stream: int
    offset: int
    stride: int
    type: int
    class_id: int

@dataclass
class VertexLayout:
    stride: int = 0
    elements: list[VDElem] = field(default_factory=list)

class PsgTemplateParser:
    RW_GRAPHICS_VERTEXDESCRIPTOR = 0x000200E9
    RW_GRAPHICS_VERTEXBUFFER = 0x000200EA
    RW_GRAPHICS_INDEXBUFFER = 0x000200EB
    PEGASUS_OPTIMESHDATA = 0x00EB0023 # This is also the RenderOptimeshData type

    def __init__(self, file_path):
        with open(file_path, 'rb') as f:
            self.data = f.read()

        self.vdes_offset = -1
        self.vertex_offset = -1
        self.face_offset = -1
        self.vbuff_dict_ptr = -1
        self.ibuff_dict_ptr = -1
        self.main_baseresource_size = 0x44
        self.graphics_baseresource_size = 0x6C
        self.vertex_buffer_size_offset = -1
        self.index_count_offset = -1
        self.optimesh_index_offset = -1

        ### NEW: Attributes to store donor skeleton info ###
        self.bone_names = []
        self.bone_palette = []

        self._parse_dictionary_and_skeleton()
        self.layout = self._parse_vdes()
        
    def _u16_be(self, offset):
        return struct.unpack('>H', self.data[offset:offset+2])[0]

    def _u32_be(self, offset):
        return struct.unpack('>I', self.data[offset:offset+4])[0]

    def _is_base_resource(self, type_id):
        return 0x00010030 <= type_id <= 0x0001003F

    ### MODIFIED: Combined parsing to find everything in one pass ###
    def _parse_dictionary_and_skeleton(self):
        try:
            num_entries = self._u32_be(0x20)
            dict_start = self._u32_be(0x30)
            main_base = self._u32_be(0x44)

            print("--- PSG Dictionary & Skeleton Parsing ---")
            print(f"Number of entries: {num_entries}")
            print(f"Dictionary starts at offset: 0x{dict_start:X}")
            print(f"Main resource base at offset: 0x{main_base:X}")

            dict_entries = []
            for i in range(num_entries):
                entry_offset = dict_start + (i * 0x18)
                entry = {
                    "ptr": self._u32_be(entry_offset + 0x00),
                    "size": self._u32_be(entry_offset + 0x08),
                    "type_id": self._u32_be(entry_offset + 0x14),
                    "offset": entry_offset
                }
                dict_entries.append(entry)

            carrier_entry = self._find_carrier(dict_entries, main_base)
            if carrier_entry:
                self._parse_carrier(carrier_entry, main_base)
            else:
                print("Warning: Could not find a valid skeleton 'Carrier' block. Skinning will not be possible.")

            palette_entry = next((e for e in dict_entries if e["type_id"] == self.PEGASUS_OPTIMESHDATA), None)
            if palette_entry:
                 self._parse_bone_palette(palette_entry, main_base)
            else:
                print("Warning: Bone palette (type 0x00EB0023) not found. Assuming identity mapping.")
                self.bone_palette = list(range(len(self.bone_names)))


            # Continue with original buffer finding logic
            for entry in dict_entries:
                type_id = entry["type_id"]
                ptr = entry["ptr"]
                block_start = (main_base + ptr) if self._is_base_resource(type_id) else ptr

                if type_id == self.RW_GRAPHICS_VERTEXDESCRIPTOR and self.vdes_offset == -1:
                    self.vdes_offset = block_start

                elif type_id == self.RW_GRAPHICS_VERTEXBUFFER and self.vertex_offset == -1:
                    br_index = self._u32_be(block_start)
                    br_entry = dict_entries[br_index]
                    br_ptr = br_entry["ptr"]
                    br_type_id = br_entry["type_id"]
                    self.vertex_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                    self.vertex_buffer_size_offset = block_start + 8
                    self.vbuff_dict_ptr = br_entry["offset"]

                elif type_id == self.RW_GRAPHICS_INDEXBUFFER and self.face_offset == -1:
                    br_index = self._u32_be(block_start)
                    br_entry = dict_entries[br_index]
                    br_ptr = br_entry["ptr"]
                    br_type_id = br_entry["type_id"]
                    self.face_offset = (main_base + br_ptr) if self._is_base_resource(br_type_id) else br_ptr
                    self.index_count_offset = block_start + 8
                    self.ibuff_dict_ptr = br_entry["offset"]
                    
                elif type_id == self.PEGASUS_OPTIMESHDATA and self.optimesh_index_offset == -1:
                    # This check is for the face count in the OptiMesh block, separate from the palette.
                    self.optimesh_index_offset = block_start + 0x64


            if self.vdes_offset == -1:
                raise ValueError("Could not find a Vertex Descriptor (0x000200E9) in the PSG template.")
            if self.vertex_offset == -1:
                raise ValueError("Could not find a Vertex Buffer (0x000200EA) in the PSG template.")
            if self.face_offset == -1:
                raise ValueError("Could not find an Index Buffer (0x000200EB) in the PSG template.")
            
            print(f"\nAuto-detected Vertex Buffer Offset: 0x{self.vertex_offset:X}")
            print(f"Auto-detected Index Buffer Offset: 0x{self.face_offset:X}")

        except (IndexError, struct.error) as e:
            raise ValueError(f"Failed to parse PSG dictionary. The template may be corrupt or invalid. Details: {e}")

    ### NEW: Logic to find and parse skeleton from the donor file ###
    def _find_carrier(self, dict_entries, main_base):
        """Finds the skeleton carrier block by checking for a valid header signature."""
        for entry in dict_entries:
            block_start = (main_base + entry["ptr"]) if self._is_base_resource(entry["type_id"]) else entry["ptr"]
            block_end = block_start + entry["size"]
            
            header_offset = block_start + 0x20
            if header_offset + 0x24 > len(self.data):
                continue

            bone_count = self._u16_be(header_offset + 0x14)
            if not (0 < bone_count <= 512):
                continue
            
            off_ibm = self._u32_be(header_offset + 0x00)
            off_tbl_idx = self._u32_be(header_offset + 0x08)
            
            ibm_abs = block_start + off_ibm
            idx_abs = block_start + off_tbl_idx
            
            if (ibm_abs + bone_count * 64 <= block_end) and (idx_abs + bone_count * 4 <= block_end):
                print(f"[Carrier] Found potential skeleton at dict entry offset 0x{entry['offset']:X} with {bone_count} bones.")
                return entry
        return None

    def _parse_carrier(self, carrier_entry, main_base):
        """Parses bone names from the found carrier block."""
        block_start = (main_base + carrier_entry["ptr"]) if self._is_base_resource(carrier_entry["type_id"]) else carrier_entry["ptr"]
        header_offset = block_start + 0x20
        
        bone_count = self._u16_be(header_offset + 0x14)
        off_tbl_idx = self._u32_be(header_offset + 0x08)
        idx_abs = block_start + off_tbl_idx
        
        print(f"\n--- Parsing Donor Skeleton ({bone_count} bones) ---")
        self.bone_names = []
        for i in range(bone_count):
            rel_offset = self._u32_be(idx_abs + 4 * i)
            name_offset = block_start + rel_offset
            
            end_offset = self.data.find(b'\x00', name_offset)
            name = self.data[name_offset:end_offset].decode('ascii', errors='ignore')
            self.bone_names.append(name)
            print(f"  Bone {i:3}: {name}")
            
    def _parse_bone_palette(self, palette_entry, main_base):
        """Parses the bone palette (map from palette index to global bone index)."""
        block_start = (main_base + palette_entry["ptr"]) if self._is_base_resource(palette_entry["type_id"]) else palette_entry["ptr"]
        palette_offset = block_start + 0x6C
        
        self.bone_palette = []
        p = palette_offset
        while p + 1 < len(self.data):
            global_index = self._u16_be(p)
            if global_index == 0xFFFF or global_index >= len(self.bone_names):
                break # Sentinel or out of bounds
            self.bone_palette.append(global_index)
            p += 2
            
        print(f"\n--- Parsing Donor Bone Palette ({len(self.bone_palette)} entries) ---")
        for i, global_idx in enumerate(self.bone_palette):
            if i < 20: # Print a sample
                 print(f"  Palette Slot {i:3} -> Global Bone {global_idx:3} ({self.bone_names[global_idx]})")
        if len(self.bone_palette) > 20:
            print("  ...")

    def _parse_vdes(self):
        header_offset = self.vdes_offset
        num_elements = struct.unpack('>H', self.data[header_offset + 10:header_offset + 12])[0]
        
        elements_offset = header_offset + 16
        parsed_elements = []
        strides = set()

        print("\n--- Vertex Descriptor Parsing ---")
        print(f"Number of elements: {num_elements}")
        print("Raw Element Data:")
        print("  Stream | Type (ID) | Vtx Type | Comps | Offset | Stride | Class")
        print("  --------------------------------------------------------------")

        for i in range(num_elements):
            elem_offset = elements_offset + (i * 8)
            elem_data = self.data[elem_offset:elem_offset+8]
            
            e = VDElem(
                vertex_type=elem_data[0],
                num_components=elem_data[1],
                stream=elem_data[2],
                offset=elem_data[3],
                stride=struct.unpack('>H', elem_data[4:6])[0],
                type=elem_data[6],
                class_id=elem_data[7]
            )
            parsed_elements.append(e)
            if e.stride > 0:
                strides.add(e.stride)
            print(f"  {e.stream:<6} | {e.type:<10} | 0x{e.vertex_type:02X}     | {e.num_components:<5} | 0x{e.offset:02X}   | {e.stride:<6} | {e.class_id}")
        
        if not strides:
            raise ValueError("Vertex descriptor in template has no valid stride defined.")
        
        resolved_stride = max(strides)
        print(f"\nDetermined vertex stride from PSG template: {resolved_stride} bytes")
        
        return VertexLayout(stride=resolved_stride, elements=parsed_elements)


class PSGConverter(QMainWindow):
    ### REMOVED: Hardcoded BONE_HIERARCHY is no longer needed ###
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.apply_stylesheet()
        self.redirect_logging()

    def on_new_text(self, text):
        """Appends text to the log window, ensuring it auto-scrolls."""
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def redirect_logging(self):
        """Redirects stdout and stderr to the in-app log window."""
        sys.stdout = Stream(newText=self.on_new_text)
        sys.stderr = Stream(newText=self.on_new_text)
        print("--- Welcome to glTF/GLB to PSG Converter ---")
        print("Ready to convert. Please select your files.\n")

    def setup_ui(self):
        self.setWindowTitle("glTF/GLB to PSG Converter (Skinned)")
        self.setMinimumSize(QSize(700, 700))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        title_label = QLabel("GLTF/GLB to PSG converter v1.4")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setObjectName("sunjayLabel")
        main_layout.insertWidget(0, title_label)
        
        credit_label = QLabel("Made by SunJay and Dumbad Enhanced by GavinSkateMan and Tuukkas")
        credit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit_label.setObjectName("creditLabel")
        main_layout.insertWidget(1, credit_label)

        input_groupbox = QGroupBox("Input Files")
        self.input_layout = QFormLayout()
        input_groupbox.setLayout(self.input_layout)

        # GLTF/GLB File
        self.entry_gltf = QLineEdit()
        browse_gltf_btn = QPushButton("Browse")
        browse_gltf_btn.clicked.connect(self.browse_gltf)
        gltf_h_layout = QHBoxLayout()
        gltf_h_layout.addWidget(self.entry_gltf)
        gltf_h_layout.addWidget(browse_gltf_btn)
        self.input_layout.addRow("glTF/GLB File:", gltf_h_layout)

        # Binary File
        self.entry_bin = QLineEdit()
        self.browse_bin_btn = QPushButton("Browse")
        self.browse_bin_btn.clicked.connect(self.browse_bin)
        self.bin_h_layout = QHBoxLayout()
        self.bin_h_layout.addWidget(self.entry_bin)
        self.bin_h_layout.addWidget(self.browse_bin_btn)
        self.bin_row_label = QLabel("Binary File (.bin):")
        self.bin_row_widget = QWidget()
        self.bin_row_widget.setLayout(self.bin_h_layout)
        self.input_layout.addRow(self.bin_row_label, self.bin_row_widget)
        self.bin_row_label.setVisible(False)
        self.bin_row_widget.setVisible(False)

        # PSG Template
        self.entry_psg = QLineEdit()
        browse_psg_btn = QPushButton("Browse")
        browse_psg_btn.clicked.connect(self.browse_psg)
        psg_h_layout = QHBoxLayout()
        psg_h_layout.addWidget(self.entry_psg)
        psg_h_layout.addWidget(browse_psg_btn)
        self.input_layout.addRow("PSG Template:", psg_h_layout)

        output_groupbox = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        output_groupbox.setLayout(output_layout)
        
        self.entry_output = QLineEdit("psg_converted.psg")
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output)
        output_h_layout = QHBoxLayout()
        output_h_layout.addWidget(self.entry_output)
        output_h_layout.addWidget(browse_output_btn)
        
        output_layout.addRow("Output File:", output_h_layout)

        self.entry_scale = QLineEdit("256.0")
        output_layout.addRow("Vertex Scale:", self.entry_scale)

        log_groupbox = QGroupBox("Conversion Log")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(5, 5, 5, 5)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setObjectName("logOutput")
        log_layout.addWidget(self.log_output)
        log_groupbox.setLayout(log_layout)
        
        self.convert_btn = QPushButton("Convert")
        self.convert_btn.setObjectName("convertButton")
        self.convert_btn.clicked.connect(self.run_conversion)
        
        self.loading_indicator = QProgressBar()
        self.loading_indicator.setRange(0, 0)
        self.loading_indicator.setTextVisible(False)
        self.loading_indicator.setVisible(False)
        
        main_layout.addWidget(input_groupbox)
        main_layout.addWidget(output_groupbox)
        main_layout.addWidget(log_groupbox)
        main_layout.addStretch(0)
        main_layout.addWidget(self.convert_btn)
        main_layout.addWidget(self.loading_indicator)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready.")

    def apply_stylesheet(self):
        stylesheet = """
            QMainWindow { background-color: #1a1a1a; color: #e0e0e0; }
            QGroupBox { font-size: 14px; font-weight: bold; border: 2px solid #333333; border-radius: 8px; margin-top: 10px; color: #e0e0e0; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 10px; background-color: #1a1a1a; }
            QLabel, QCheckBox { font-size: 12px; color: #e0e0e0; }
            QLabel#sunjayLabel { font-size: 24px; font-weight: bold; padding: 10px; border: 2px solid #6c5ce7; border-radius: 10px; background-color: #2b2b2b; }
            QLabel#creditLabel { font-size: 10px; color: #999999; }
            QLineEdit, QComboBox { border: 1px solid #444444; border-radius: 5px; padding: 5px 8px; background-color: #2b2b2b; color: #e0e0e0; }
            QLineEdit:focus, QComboBox:focus { border: 1px solid #6c5ce7; }
            QPushButton { background-color: #333333; color: #e0e0e0; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #444444; }
            QPushButton#convertButton { background-color: #6c5ce7; font-size: 16px; padding: 15px 30px; }
            QPushButton#convertButton:hover { background-color: #8a7ceb; }
            QPushButton:disabled { background-color: #4a4aa4a; color: #888888; }
            QStatusBar { background-color: #2b2b2b; color: #e0e0e0; border-top: 1px solid #444444; padding: 5px; }
            QTextEdit#logOutput {
                background-color: #222222;
                color: #cccccc;
                border: 1px solid #444444;
                border-radius: 5px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 11px;
            }
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                height: 10px;
            }
            QProgressBar::chunk {
                background-color: #6c5ce7;
                width: 15px;
                margin: 0.5px;
            }
        """
        self.setStyleSheet(stylesheet)
    
    def browse_gltf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select glTF/GLB File", "", "glTF Files (*.gltf *.glb)")
        if path:
            self.entry_gltf.setText(path)
            if path.lower().endswith('.gltf'):
                self.bin_row_label.setVisible(True)
                self.bin_row_widget.setVisible(True)
                try:
                    gltf = pygltflib.GLTF2.load(path)
                    if gltf.buffers and gltf.buffers[0].uri:
                        bin_uri = gltf.buffers[0].uri
                        bin_path = os.path.join(os.path.dirname(path), bin_uri)
                        if os.path.exists(bin_path):
                            self.entry_bin.setText(bin_path)
                        else:
                            self.entry_bin.setText("")
                except Exception:
                    self.entry_bin.setText("")
            else:
                self.bin_row_label.setVisible(False)
                self.bin_row_widget.setVisible(False)
                self.entry_bin.setText("")

    def browse_bin(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Binary File", "", "Binary Files (*.bin)")
        if path: self.entry_bin.setText(path)

    def browse_psg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PSG Template File", "", "PSG Files (*.psg)")
        if path: self.entry_psg.setText(path)

    def browse_output(self):
        initial_file = self.entry_output.text() or "psg_converted.psg"
        path, _ = QFileDialog.getSaveFileName(self, "Save Converted PSG File", initial_file, "PSG Files (*.psg)")
        if path: self.entry_output.setText(path)
            
    ### REMOVED: get_sorted_skeleton is no longer needed ###

    @staticmethod
    def normalize_bone_name(name: str | None) -> str | None:
        if name is None:
            return None
        # Remove all non-alphanumerics and lowercase to create a robust canonical form
        return ''.join(ch for ch in name if ch.isalnum()).lower()

    def run_conversion(self):
        self.log_output.clear()
        print("--- Starting Conversion Process ---\n")
        
        gltf_path = self.entry_gltf.text()
        bin_path = self.entry_bin.text()
        psg_template_path = self.entry_psg.text()
        output_path = self.entry_output.text()
        try:
            scale_xyz = float(self.entry_scale.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Vertex Scale must be a valid number.")
            print("ERROR: Vertex Scale input is not a valid number.")
            return

        if not all([gltf_path, psg_template_path, output_path]):
            QMessageBox.critical(self, "Error", "Please specify all required files (glTF, PSG Template, Output).")
            print("ERROR: One or more required file paths are missing.")
            return
        
        if gltf_path.lower().endswith('.gltf') and not bin_path:
            QMessageBox.critical(self, "Error", "A .bin file is required when using a .gltf file.")
            print("ERROR: A .bin file is required for .gltf input.")
            return

        self.convert_btn.setEnabled(False)
        self.loading_indicator.setVisible(True)
        self.statusBar.showMessage("Converting... Please wait")
        QApplication.processEvents()
        
        try:
            print(f"Parsing PSG Template: {os.path.basename(psg_template_path)}")
            template = PsgTemplateParser(psg_template_path)
            print("PSG Template parsed successfully.\n")

            print(f"Loading glTF/GLB file: {os.path.basename(gltf_path)}")
            model_data = self.parse_gltf_to_data(gltf_path, bin_path)
            (final_vertices, final_uvs, final_normals, final_tangents, final_binormals, 
             final_faces, final_joints, final_weights, glb_bone_map) = model_data
            
            print("--- GLTF Data Loaded ---")
            print(f"Total Vertices: {len(final_vertices)}")
            print(f"Total Faces: {len(final_faces)}")
            is_skinned = final_joints is not None
            if is_skinned:
                if not template.bone_names or not template.bone_palette:
                    raise ValueError("Skinning data found in GLB, but no skeleton or palette was loaded from the donor PSG.")
                print(f"Skinned model detected with {len(glb_bone_map)} bones.")
            else:
                print("No skinning data found in GLB. Processing as a static mesh.")
            print("\n")
            
            remapped_joints, remapped_weights = None, None
            if is_skinned:
                print("--- Bone & Skin Remapping (Donor-based) ---")
                
                ### MODIFIED: Call new remapping function ###
                remapped_joints, remapped_weights = self.remap_skin_to_donor_palette(
                    final_joints, final_weights, glb_bone_map, 
                    template.bone_names, template.bone_palette)
                
                print("Bone remapping complete.\n")
                
                print("--- Sample Vertex Data (First 10) ---")
                for i in range(min(10, len(final_vertices))):
                    print(f"Vertex {i}:")
                    print(f"  Position:  ({final_vertices[i][0]:.3f}, {final_vertices[i][1]:.3f}, {final_vertices[i][2]:.3f})")
                    print(f"  Normal:    ({final_normals[i][0]:.3f}, {final_normals[i][1]:.3f}, {final_normals[i][2]:.3f})")
                    print(f"  UVs:       ({final_uvs[i][0]:.3f}, {final_uvs[i][1]:.3f})")
                    raw_j_names = [glb_bone_map.get(idx, 'N/A') for idx in final_joints[i]]
                    print(f"  Raw Joints:  {list(final_joints[i])} -> Names: {raw_j_names}")
                    print(f"  Raw Weights: {[f'{w:.3f}' for w in final_weights[i]]}")
                    print(f"  Remapped Palette Joints: {list(remapped_joints[i])}")
                    print(f"  Remapped Final Weights: {[f'{w:.3f}' for w in remapped_weights[i]]}")
                    print("-----------------------------")


            print("--- Generating Binary Data Blocks ---")
            vertex_data = self.make_vertex_bin_dynamic(
                final_vertices, final_uvs, final_normals, final_tangents, final_binormals,
                remapped_joints, remapped_weights, template.layout, scale_xyz=float(self.entry_scale.text()))
            face_data = self.make_face_bin(final_faces)
            print(f"Vertex block size: {len(vertex_data)} bytes")
            print(f"Face block size: {len(face_data)} bytes\n")
            
            print("--- Assembling Final PSG File ---")
            with open(psg_template_path, 'rb') as f:
                psg_data = bytearray(f.read())
                print(f"Read {len(psg_data)} bytes from template.")

            v_offset = template.vertex_offset
            
            original_file_end = struct.unpack(">I", psg_data[template.main_baseresource_size:template.main_baseresource_size+4])[0]
            psg_data = psg_data[0:original_file_end]
            
            psg_data[template.graphics_baseresource_size:template.graphics_baseresource_size+4] = struct.pack(">I", len(vertex_data) + len(face_data))
            psg_data[template.vertex_buffer_size_offset:template.vertex_buffer_size_offset+4] = struct.pack(">I", len(vertex_data))
            psg_data[template.index_count_offset:template.index_count_offset+4] = struct.pack(">I", len(final_faces) * 3)
            if template.optimesh_index_offset > 0:
                psg_data[template.optimesh_index_offset:template.optimesh_index_offset+4] = struct.pack(">I", len(final_faces) * 3)

            psg_data.extend(b'\x00' * (len(vertex_data) + len(face_data)))
            psg_data[v_offset:v_offset + len(vertex_data)] = vertex_data
            psg_data[template.vbuff_dict_ptr+8:template.vbuff_dict_ptr+12] = struct.pack(">I", len(vertex_data))
            
            new_f_offset = v_offset + len(vertex_data)
            psg_data[template.ibuff_dict_ptr:template.ibuff_dict_ptr+4] = struct.pack(">I", len(vertex_data))
            psg_data[template.ibuff_dict_ptr+8:template.ibuff_dict_ptr+12] = struct.pack(">I", len(face_data))
            psg_data[new_f_offset:new_f_offset + len(face_data)] = face_data

            print(f"Final PSG size: {len(psg_data)} bytes")
            print(f"Writing output to: {output_path}")
            with open(output_path, 'wb') as f:
                f.write(psg_data)
            
            print("\n--- CONVERSION SUCCESSFUL ---")
            self.statusBar.showMessage(f"Conversion successful! Saved to {output_path}")
            QMessageBox.information(self, "Success", f"glTF successfully converted to PSG!\n\nVertices: {len(final_vertices)}\nFaces: {len(final_faces)}\nOutput: {output_path}")
        
        except Exception as e:
            self.statusBar.showMessage(f"Error: {str(e)}")
            print("\n--- CONVERSION ERROR ---")
            traceback.print_exc()
            print("------------------------")
            QMessageBox.critical(self, "Error", f"An error occurred during conversion. Please check the log window for details.\n\nError: {str(e)}")
        finally:
            self.convert_btn.setEnabled(True)
            self.loading_indicator.setVisible(False)

    ### CORRECTED LOGIC TO HANDLE DUPLICATE BONE PALETTE ENTRIES ###
    def remap_skin_to_donor_palette(self, gltf_joints, gltf_weights, glb_bone_map, donor_bone_names, donor_bone_palette):
        # 1. Create lookup maps from donor data
        donor_name_to_global_idx = {self.normalize_bone_name(name): i for i, name in enumerate(donor_bone_names)}
        
        # --- START: CORRECTED CODE ---
        # This new logic correctly handles cases where a bone appears multiple times
        # in the donor's palette by only mapping to the FIRST occurrence.
        global_idx_to_palette_idx = {}
        for palette_idx, global_idx in enumerate(donor_bone_palette):
            if global_idx not in global_idx_to_palette_idx:
                global_idx_to_palette_idx[global_idx] = palette_idx
        # --- END: CORRECTED CODE ---

        # 2. Create a map from GLTF joint index -> Donor Palette Index
        gltf_to_palette_map = {}
        unmapped_bones = set()
        print("--- GLTF to Donor Palette Bone Index Mapping ---")
        print("glTF Bone Name       -> glTF Idx -> Donor Global Idx -> Donor Palette Idx")
        for gltf_idx, gltf_name in glb_bone_map.items():
            norm_name = self.normalize_bone_name(gltf_name)
            
            global_idx = donor_name_to_global_idx.get(norm_name)
            palette_idx = global_idx_to_palette_idx.get(global_idx) if global_idx is not None else None
            
            if palette_idx is not None:
                gltf_to_palette_map[gltf_idx] = palette_idx
            else:
                 unmapped_bones.add(gltf_name)

            print(f"{str(gltf_name):<20} -> {gltf_idx:<9} -> {str(global_idx):<17} -> {str(palette_idx)}")
        print("---------------------------------------------------------------------")

        if unmapped_bones:
            print(f"\nWarning: {len(unmapped_bones)} bones from the GLTF file were not found in the donor's skeleton/palette and were ignored:")
            for bone in sorted(list(unmapped_bones)):
                print(f" - {bone}")
        
        # 3. Process vertices
        final_palette_indices = []
        final_weights = []

        for indices, weights in zip(gltf_joints, gltf_weights):
            weight_by_palette_idx = {}
            for i in range(4):
                w = float(weights[i])
                if w <= 1e-6:
                    continue
                
                gltf_joint_idx = int(indices[i])
                palette_idx = gltf_to_palette_map.get(gltf_joint_idx)
                
                if palette_idx is not None:
                    weight_by_palette_idx[palette_idx] = weight_by_palette_idx.get(palette_idx, 0.0) + w

            # Sort by aggregated weight and take top 4
            sorted_pairs = sorted(weight_by_palette_idx.items(), key=lambda x: x[1], reverse=True)[:4]
            
            palette_indices_per_vertex = [0] * 4
            weights_per_vertex = [0.0] * 4

            for i, (pal_idx, w) in enumerate(sorted_pairs):
                palette_indices_per_vertex[i] = int(pal_idx)
                weights_per_vertex[i] = float(w)

            # Normalize final weights
            total_weight = sum(weights_per_vertex)
            if total_weight > 1e-6:
                inv = 1.0 / total_weight
                weights_per_vertex = [w * inv for w in weights_per_vertex]
            else:
                # Fallback: full weight to palette index 0 (usually root)
                palette_indices_per_vertex = [0, 0, 0, 0]
                weights_per_vertex = [1.0, 0.0, 0.0, 0.0]

            final_palette_indices.append(palette_indices_per_vertex)
            final_weights.append(weights_per_vertex)

        return numpy.array(final_palette_indices, dtype=numpy.uint8), numpy.array(final_weights, dtype=numpy.float32)

    def pack_normal_dec3n(self, n):
        """
        Packs a normal vector into a signed 11:11:10 32-bit integer.
        The function name is kept for compatibility with the calling code,
        but the format has been changed from DEC3N to CMP 11:11:10 signed.
        """
        nx, ny, nz = n

        # Clamp the input floats to the [-1.0, 1.0] range
        nx = max(-1.0, min(1.0, nx))
        ny = max(-1.0, min(1.0, ny))
        nz = max(-1.0, min(1.0, nz))

        # Scale to the integer range for each component
        # X: 11-bit signed -> max value is 2^(11-1) - 1 = 1023
        # Y: 11-bit signed -> max value is 2^(11-1) - 1 = 1023
        # Z: 10-bit signed -> max value is 2^(10-1) - 1 = 511
        ix = int(round(nx * 1023.0))
        iy = int(round(ny * 1023.0))
        iz = int(round(nz * 511.0))

        # Apply bitmasks to handle two's complement representation correctly
        # for the respective bit widths.
        mask_x = (1 << 11) - 1  # 0x7FF
        mask_y = (1 << 11) - 1  # 0x7FF
        mask_z = (1 << 10) - 1  # 0x3FF

        ix &= mask_x
        iy &= mask_y
        iz &= mask_z

        # Combine into a single 32-bit integer with layout:
        # | Z (10 bits) | Y (11 bits) | X (11 bits) |
        packed_val = (iz << 22) | (iy << 11) | ix
        
        # Pack as a big-endian unsigned integer
        return struct.pack('>I', packed_val)

    def make_vertex_bin_dynamic(self, vertices, uvs, normals, tangents, binormals, joints, weights, layout: VertexLayout, scale_xyz=256):
        output = bytearray()
        elem_map = {
            'XYZ': 0, 'WEIGHTS': 1, 'NORMAL': 2, 'VERTEXCOLOR': 3, 'SPECULAR': 4,
            'BONEINDICES': 7, 'TEX0': 8, 'TEX1': 9, 'TEX2': 10, 'TEX3': 11, 'TEX4': 12, 'TEX5': 13,
            'TANGENT': 14, 'BINORMAL': 15
        }
        is_skinned = joints is not None and weights is not None

        for i in range(len(vertices)):
            vertex_bytes = bytearray(layout.stride)
            for elem in layout.elements:
                packed_data = b''
                if elem.type == elem_map['XYZ']:
                    x_s, y_s, z_s = [max(-32768, min(32767, int(c * scale_xyz))) for c in vertices[i]]
                    if elem.vertex_type in [0x01, 0x05]: packed_data = struct.pack('>hhh', x_s, y_s, z_s)
                    elif elem.vertex_type == 0x02: packed_data = struct.pack('>fff', *vertices[i])
                elif elem.type == elem_map['NORMAL']:
                    if elem.vertex_type == 0x06: packed_data = self.pack_normal_dec3n(normals[i])
                elif elem.type == elem_map['TANGENT']:
                     if elem.vertex_type == 0x06: packed_data = self.pack_normal_dec3n(tangents[i])
                elif elem.type == elem_map['BINORMAL']:
                     if elem.vertex_type == 0x06: packed_data = self.pack_normal_dec3n(binormals[i])
                elif elem.type == elem_map['TEX0']:
                    u, v = uvs[i]
                    if elem.vertex_type == 0x03: packed_data = struct.pack('>ee', numpy.float16(u), numpy.float16(v)) # Correct packing for half-float
                    elif elem.vertex_type in [0x01, 0x05]:
                        u_s, v_s = [max(-32768, min(32767, int(round(c * 32767.0)))) for c in (u, v)]
                        packed_data = struct.pack('>hh', u_s, v_s)
                elif elem.type == elem_map['WEIGHTS']:
                    w = weights[i] if is_skinned else [1.0, 0.0, 0.0, 0.0]
                    if elem.vertex_type == 0x02:
                        packed_data = struct.pack('>ffff', *w)
                    elif elem.vertex_type in [0x04, 0x07]:
                        w_u8 = [int(round(max(0.0, min(1.0, float(c))) * 255.0)) for c in w]
                        packed_data = struct.pack('>BBBB', *w_u8)
                elif elem.type == elem_map['BONEINDICES']:
                    j = joints[i] if is_skinned else [0, 0, 0, 0]
                    if elem.vertex_type in [0x04, 0x07]:
                        j_u8 = [int(max(0, min(255, int(v)))) for v in j]
                        packed_data = struct.pack('>BBBB', *j_u8)
                elif elem.type == elem_map['VERTEXCOLOR']:
                    if elem.vertex_type in [0x04, 0x07]: packed_data = struct.pack('>BBBB', 255, 255, 255, 255)
                elif elem.type == elem_map['SPECULAR']:
                    if elem.vertex_type in [0x04, 0x07]: packed_data = struct.pack('>BBBB', 0, 0, 0, 255)
                if packed_data:
                    vertex_bytes[elem.offset:elem.offset + len(packed_data)] = packed_data
            output.extend(vertex_bytes)
        return output

    def make_face_bin(self, faces):
        output = bytearray()
        for face in faces:
            for idx in face:
                output.extend(struct.pack('>H', idx))
        return output

    def parse_gltf_to_data(self, gltf_path, bin_path):
        gltf = pygltflib.GLTF2.load(gltf_path)
        
        blob = None
        if gltf_path.lower().endswith('.glb'): blob = gltf.binary_blob()
        elif os.path.exists(bin_path):
            with open(bin_path, 'rb') as f: blob = f.read()

        if blob is None: raise ValueError("Could not load binary data.")
        if not gltf.meshes: raise ValueError("No meshes found in file.")
        if len(gltf.meshes) > 1 or len(gltf.meshes[0].primitives) > 1:
            print("Warning: Model has multiple parts. Using first primitive of first mesh.")
            
        primitive = gltf.meshes[0].primitives[0]
        
        def get_accessor_data(accessor_id):
            accessor = gltf.accessors[accessor_id]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
            dtype_map = {5120: numpy.int8, 5121: numpy.uint8, 5122: numpy.int16, 5123: numpy.uint16, 5125: numpy.uint32, 5126: numpy.float32}
            dtype = dtype_map[accessor.componentType]
            num_components = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[accessor.type]
            data = numpy.frombuffer(blob, dtype=dtype, count=accessor.count * num_components, offset=offset)
            return data.reshape(accessor.count, num_components) if num_components > 1 else data

        raw_vertices = get_accessor_data(primitive.attributes.POSITION)
        raw_normals = get_accessor_data(primitive.attributes.NORMAL)
        raw_uvs = get_accessor_data(primitive.attributes.TEXCOORD_0) if primitive.attributes.TEXCOORD_0 is not None else numpy.zeros((len(raw_vertices), 2), dtype=numpy.float32)
        indices = get_accessor_data(primitive.indices)
        faces_indices = indices.reshape(-1, 3)

        raw_joints, raw_weights, glb_bone_map = None, None, None
        
        # ### FIXED ###
        # This check is more robust. It verifies the attributes exist and are not None,
        # which prevents the TypeError on models without bones.
        if primitive.attributes.JOINTS_0 is not None and primitive.attributes.WEIGHTS_0 is not None:
            raw_joints = get_accessor_data(primitive.attributes.JOINTS_0)
            raw_weights = get_accessor_data(primitive.attributes.WEIGHTS_0)

            weights_accessor = gltf.accessors[primitive.attributes.WEIGHTS_0]
            if weights_accessor.componentType == 5121: # UBYTE
                raw_weights = raw_weights.astype(numpy.float32) / 255.0
            elif weights_accessor.componentType == 5123: # USHORT
                raw_weights = raw_weights.astype(numpy.float32) / 65535.0

            skin_index = None
            for node in gltf.nodes:
                if node.mesh == 0 and node.skin is not None:
                    skin_index = node.skin
                    break
            
            if skin_index is None:
                 raise ValueError("Skinned mesh data found, but no node in the GLB uses this mesh with a skin.")

            if gltf.skins and len(gltf.skins) > skin_index:
                skin = gltf.skins[skin_index]
                glb_bone_map = {i: gltf.nodes[joint_index].name for i, joint_index in enumerate(skin.joints)}
            else:
                raise ValueError("Skinned data found, but no valid skin definition was found in the GLB.")

        tangent_acc = numpy.zeros_like(raw_vertices)
        for i0, i1, i2 in faces_indices:
            p0, p1, p2 = raw_vertices[[i0, i1, i2]]
            uv0, uv1, uv2 = raw_uvs[[i0, i1, i2]]
            edge1, edge2 = p1 - p0, p2 - p0
            delta_uv1, delta_uv2 = uv1 - uv0, uv2 - uv0
            f = delta_uv1[0] * delta_uv2[1] - delta_uv2[0] * delta_uv1[1]
            if abs(f) > 1e-6:
                r = 1.0 / f
                tangent = (edge1 * delta_uv2[1] - edge2 * delta_uv1[1]) * r
                tangent_acc[[i0, i1, i2]] += tangent
        
        t_ortho = tangent_acc - raw_normals * numpy.sum(tangent_acc * raw_normals, axis=1, keepdims=True)
        final_raw_tangents = t_ortho / (numpy.linalg.norm(t_ortho, axis=1, keepdims=True) + 1e-9)
        final_raw_binormals = numpy.cross(raw_normals, final_raw_tangents)

        final_data = { "vertices": [], "uvs": [], "normals": [], "tangents": [], "binormals": [], "joints": [], "weights": [] }
        is_skinned = raw_joints is not None

        for v_idx in indices:
            final_data["vertices"].append(raw_vertices[v_idx])
            final_data["normals"].append(raw_normals[v_idx])
            final_data["uvs"].append(raw_uvs[v_idx])
            final_data["tangents"].append(final_raw_tangents[v_idx])
            final_data["binormals"].append(final_raw_binormals[v_idx])
            if is_skinned:
                final_data["joints"].append(raw_joints[v_idx])
                final_data["weights"].append(raw_weights[v_idx])

        final_faces = numpy.arange(len(indices)).reshape(-1, 3).tolist()
        joints_out = final_data["joints"] if is_skinned else None
        weights_out = final_data["weights"] if is_skinned else None

        return (final_data["vertices"], final_data["uvs"], final_data["normals"], final_data["tangents"],
                final_data["binormals"], final_faces, joints_out, weights_out, glb_bone_map)

if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        converter = PSGConverter()
        converter.show()
        sys.exit(app.exec())
    except Exception as e:
        print("An unexpected error occurred:")
        traceback.print_exc()
        input("Press Enter to exit...")