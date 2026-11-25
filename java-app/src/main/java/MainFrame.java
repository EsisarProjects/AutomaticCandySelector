import javax.swing.*;
import javax.swing.border.LineBorder;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class MainFrame extends JFrame {
    private JLabel dropZoneLabel;
    private JTextArea logArea;
    private JTextArea resultInfoArea;

    // Optional environment overrides: ACS_PROJECT_ROOT and ACS_PYTHON.
    private static File projectRoot() {
        String override = System.getenv("ACS_PROJECT_ROOT");
        File dir = override != null && !override.isEmpty()
            ? new File(override) : new File(System.getProperty("user.dir"));
        dir = dir.getAbsoluteFile();
        for (int i = 0; i < 5 && dir != null; i++, dir = dir.getParentFile()) {
            if (new File(dir, "worker.py").isFile() &&
                new File(dir, "configs/application.yaml").isFile()) {
                return dir;
            }
        }
        throw new IllegalStateException("ACS project root not found. Set ACS_PROJECT_ROOT.");
    }

    // Quản lý tiến trình Python chạy nền liên tục
    private Process pythonProcess;
    private BufferedWriter pythonInput;
    private BufferedReader pythonOutput;
    private volatile boolean isPythonReady = false;
    private String lastResultImagePath = null;

    public MainFrame() {
        setTitle("ACS - Automatic Candy Selector (Real-time Pipeline)");
        setSize(1150, 780);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        // --- BẢNG MÀU GIAO DIỆN HIỆN ĐẠI (DARK THEME) ---
        Color bgDark = new Color(18, 18, 18);
        Color panelDark = new Color(30, 30, 30);
        Color accentOrange = new Color(255, 111, 0);
        Color textLight = new Color(220, 220, 220);

        getContentPane().setBackground(bgDark);
        setLayout(new BorderLayout(10, 10));

        // --- HEADER ---
        JPanel headerPanel = new JPanel(new BorderLayout());
        headerPanel.setBackground(panelDark);
        headerPanel.setBorder(BorderFactory.createEmptyBorder(15, 20, 15, 20));
        
        JLabel titleLabel = new JLabel("ACS | Automatic Candy Selector");
        titleLabel.setFont(new Font("Segoe UI", Font.BOLD, 22));
        titleLabel.setForeground(accentOrange);
        headerPanel.add(titleLabel, BorderLayout.WEST);

        JButton selectBtn = new JButton("📂 Chọn ảnh từ máy");
        selectBtn.setBackground(accentOrange);
        selectBtn.setForeground(Color.WHITE);
        selectBtn.setFont(new Font("Segoe UI", Font.BOLD, 13));
        selectBtn.setFocusPainted(false);
        selectBtn.setCursor(new Cursor(Cursor.HAND_CURSOR));
        headerPanel.add(selectBtn, BorderLayout.EAST);
        add(headerPanel, BorderLayout.NORTH);

        // --- VÙNG TRUNG TÂM (ẢNH KẾT QUẢ & BẢNG THÔNG TIN BÊN CẠNH) ---
        JPanel centerPanel = new JPanel(new BorderLayout(15, 0));
        centerPanel.setBackground(bgDark);
        centerPanel.setBorder(BorderFactory.createEmptyBorder(0, 15, 0, 15));

        // 1. Bên trái: Vùng Drop Zone kiêm hiển thị ảnh kết quả AI
        JPanel imageContainerPanel = new JPanel(new BorderLayout());
        imageContainerPanel.setBackground(panelDark);
        imageContainerPanel.setBorder(new LineBorder(new Color(65, 65, 65), 2, true));
        
        dropZoneLabel = new JLabel("<html><div style='text-align: center;'>📁<br><br><b>KÉO & THẢ ẢNH VÀO ĐÂY</b><br><span style='font-size:11px; color:gray;'>Hỗ trợ JPG, PNG, WEBP</span></div></html>", JLabel.CENTER);
        dropZoneLabel.setForeground(textLight);
        dropZoneLabel.setFont(new Font("Segoe UI", Font.PLAIN, 16));
        imageContainerPanel.add(dropZoneLabel, BorderLayout.CENTER);
        centerPanel.add(imageContainerPanel, BorderLayout.CENTER);

        // 2. Bên phải: Bảng thông tin nhỏ gọn (Tên kẹo & tỉ lệ)
        JPanel infoPanel = new JPanel(new BorderLayout());
        infoPanel.setBackground(panelDark);
        infoPanel.setBorder(BorderFactory.createTitledBorder(
            new LineBorder(new Color(65, 65, 65), 1, true),
            " Chi tiết nhận diện ", 
            0, 0, new Font("Segoe UI", Font.BOLD, 12), accentOrange
        ));

        resultInfoArea = new JTextArea();
        resultInfoArea.setEditable(false);
        resultInfoArea.setBackground(new Color(24, 24, 24));
        resultInfoArea.setForeground(new Color(240, 240, 240));
        resultInfoArea.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        resultInfoArea.setText("\n  Chưa có ảnh nào được xử lý.\n  Vui lòng chọn hoặc kéo thả ảnh\n  vào khung bên trái.");
        
        JScrollPane infoScroll = new JScrollPane(resultInfoArea);
        infoScroll.setBorder(BorderFactory.createEmptyBorder());
        infoPanel.add(infoScroll, BorderLayout.CENTER);
        
        // Cố định chiều rộng panel bên phải
        infoPanel.setPreferredSize(new Dimension(340, 0));
        centerPanel.add(infoPanel, BorderLayout.EAST);

        add(centerPanel, BorderLayout.CENTER);

        // --- DƯỚI CÙNG: KHUNG NHẬT KÝ HỆ THỐNG ---
        logArea = new JTextArea(5, 40);
        logArea.setEditable(false);
        logArea.setBackground(new Color(12, 12, 12));
        logArea.setForeground(new Color(0, 255, 128));
        logArea.setFont(new Font("Consolas", Font.PLAIN, 13));
        JScrollPane logScroll = new JScrollPane(logArea);
        logScroll.setBorder(BorderFactory.createTitledBorder(null, " Nhật ký hoạt động ", 0, 0, new Font("Segoe UI", Font.BOLD, 11), accentOrange));
        add(logScroll, BorderLayout.SOUTH);

        // --- CƠ CHẾ KÉO & THẢ (DRAG & DROP) ---
        new DropTarget(imageContainerPanel, new DropTargetListener() {
            public void dragEnter(DropTargetDragEvent dtde) {}
            public void dragOver(DropTargetDragEvent dtde) {}
            public void dropActionChanged(DropTargetDragEvent dtde) {}
            public void dragExit(DropTargetEvent dte) {}
            
            @SuppressWarnings("unchecked")
            public void drop(DropTargetDropEvent dtde) {
                try {
                    dtde.acceptDrop(DnDConstants.ACTION_COPY);
                    Transferable tr = dtde.getTransferable();
                    if (tr.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
                        List<File> fileList = (List<File>) tr.getTransferData(DataFlavor.javaFileListFlavor);
                        if (!fileList.isEmpty()) {
                            processSelectedFile(fileList.get(0));
                        }
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }
        });

        // --- SỰ KIỆN NÚT CHỌN ẢNH ---
        selectBtn.addActionListener(e -> {
            JFileChooser chooser = new JFileChooser();
            if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                processSelectedFile(chooser.getSelectedFile());
            }
        });

        // Tắt tiến trình Python ngầm khi đóng ứng dụng Java
        Runtime.getRuntime().addShutdownHook(new Thread(this::stopPythonWorker));

        // Khởi động AI chạy nền ngay khi mở App
        new Thread(this::initPythonWorker).start();
    }

    private void processSelectedFile(File file) {
        // Hiển thị tạm ảnh gốc lên khung trái trong lúc chờ AI xử lý
        showImageOnDropZone(file.getAbsolutePath());
        if (!isPythonReady) {
            logArea.append("⚠️ AI đang nạp Model vào RAM, vui lòng chờ vài giây...\n");
            return;
        }
        sendImageToAI(file.getAbsolutePath());
    }

    private void showImageOnDropZone(String imagePath) {
        ImageIcon icon = new ImageIcon(imagePath);
        // Tự động căn chỉnh kích thước ảnh vừa vặn với khung hiển thị (khoảng 750x520)
        Image img = icon.getImage().getScaledInstance(750, 520, Image.SCALE_SMOOTH);
        dropZoneLabel.setIcon(new ImageIcon(img));
        dropZoneLabel.setText("");
    }

    // 1. Khởi tạo Python Worker ngầm
    private void initPythonWorker() {
        try {
            SwingUtilities.invokeLater(() -> logArea.append("Đang kết nối và nạp mô hình AI vào bộ nhớ...\n"));

            String python = System.getenv("ACS_PYTHON");
            if (python == null || python.isEmpty()) python = "python";
            ProcessBuilder pb = new ProcessBuilder(python, "-u", "worker.py");
            pb.directory(projectRoot());
            pb.environment().put("PYTHONIOENCODING", "utf-8");
            pb.redirectErrorStream(true);

            pythonProcess = pb.start();
            pythonInput = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream(), StandardCharsets.UTF_8));
            pythonOutput = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream(), StandardCharsets.UTF_8));

            String line;
            while ((line = pythonOutput.readLine()) != null) {
                final String l = line;
                SwingUtilities.invokeLater(() -> logArea.append(l + "\n"));
                if (l.trim().equals("READY")) {
                    isPythonReady = true;
                    SwingUtilities.invokeLater(() -> logArea.append(">>> AI ĐÃ SẴN SÀNG! Bạn có thể kéo thả ảnh ngay bây giờ.\n\n"));
                    break;
                }
            }

            // Luồng đọc log và đường dẫn ảnh kết quả liên tục
            new Thread(() -> {
                try {
                    String logLine;
                    while ((logLine = pythonOutput.readLine()) != null) {
                        final String str = logLine;
                        if (str.startsWith("RESULT:")) {
                            lastResultImagePath = str.substring(7).trim();
                        } else if (str.trim().equals("DONE")) {
                            if (lastResultImagePath != null) {
                                String resultPath = lastResultImagePath;
                                lastResultImagePath = null;
                                SwingUtilities.invokeLater(() -> {
                                    displayResultImage(resultPath);
                                    loadAndDisplayJsonResult(resultPath);
                                });
                            }
                        } else {
                            SwingUtilities.invokeLater(() -> logArea.append(str + "\n"));
                        }
                    }
                } catch (IOException ignored) {}
            }).start();

        } catch (Exception ex) {
            ex.printStackTrace();
            SwingUtilities.invokeLater(() -> logArea.append("Lỗi khởi động AI: " + ex.getMessage() + "\n"));
        }
    }

    private void sendImageToAI(String imagePath) {
        try {
            lastResultImagePath = null;
            logArea.append(">> Bắt đầu nhận diện: " + new File(imagePath).getName() + "...\n");
            pythonInput.write(imagePath);
            pythonInput.newLine();
            pythonInput.flush();
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void displayResultImage(String imagePath) {
        File f = new File(imagePath);
        if (f.exists()) {
            showImageOnDropZone(f.getAbsolutePath());
            logArea.append(">>> HOÀN TẤT KẾT QUẢ!\n\n");
        }
    }

    // Đọc file JSON kết quả do Python sinh ra để hiển thị danh sách kẹo bên bảng phải
    private void loadAndDisplayJsonResult(String resultImagePath) {
        try {
            // File JSON kết quả có dạng: [ten_anh]_[hash]_predictions.json
            // Nằm cùng thư mục với ảnh kết quả
            File imgFile = new File(resultImagePath);
            String parentDir = imgFile.getParent();
            String imageName = imgFile.getName();
            
            // Cắt đuôi _annote.jpg để lấy phần stem
            if (imageName.endsWith("_annote.jpg")) {
                String stem = imageName.substring(0, imageName.length() - 11);
                File jsonFile = new File(parentDir, stem + "_predictions.json");
                
                if (jsonFile.exists()) {
                    String content = new String(Files.readAllBytes(jsonFile.toPath()), StandardCharsets.UTF_8);
                    parseAndShowJsonData(content);
                }
            }
        } catch (Exception e) {
            resultInfoArea.setText("\n  Không thể đọc file chi tiết kết quả.");
        }
    }

    // Phân tích dữ liệu JSON thủ công (dùng Regex nhẹ nhàng không cần thư viện ngoài)
    private void parseAndShowJsonData(String jsonContent) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n  DANH SÁCH KẸO NHẬN DIỆN:\n");
        sb.append("  -----------------------------------\n");

        try {
            // Tách các mục detections bằng cách split cơ bản
            if (jsonContent.contains("detections")) {
                int count = 0;
                String[] items = jsonContent.split("\\{\\s*\"id\":");
                for (int i = 1; i < items.length; i++) {
                    String item = items[i];
                    count++;

                    // Lấy type_predit
                    String candyType = extractJsonField(item, "type_predit");
                    // Lấy couleur_predite
                    String candyColor = extractJsonField(item, "couleur_predite");
                    // Lấy confiance_type
                    String confType = extractJsonField(item, "confiance_type");

                    double conf = 0.0;
                    try { conf = Double.parseDouble(confType); } catch(Exception ignored) {}

                    sb.append(String.format("  [%02d] %s / %s\n", count, candyType, candyColor));
                    sb.append(String.format("       Độ tin cậy: %.1f%%\n", conf * 100));
                    sb.append("  -----------------------------------\n");
                }
                sb.insert(0, String.format("  Tổng số lượng: %d viên\n", count));
            }
        } catch (Exception e) {
            sb.append("  Lỗi phân tích dữ liệu JSON.");
        }

        resultInfoArea.setText(sb.toString());
    }

    private String extractJsonField(String json, String field) {
        try {
            String searchKey = "\"" + field + "\":";
            int idx = json.indexOf(searchKey);
            if (idx == -1) {
                searchKey = "\"" + field + "\" :";
                idx = json.indexOf(searchKey);
            }
            if (idx == -1) return "0";

            int colon = json.indexOf(":", idx);
            int comma = json.indexOf(",", colon);
            int brace = json.indexOf("}", colon);
            
            int end = comma;
            if (comma == -1 || (brace != -1 && brace < comma)) {
                end = brace;
            }
            if (end == -1) return "0";

            String val = json.substring(colon + 1, end).trim();
            // Xóa dấu nháy kép nếu có
            val = val.replace("\"", "");
            return val;
        } catch (Exception e) {
            return "0";
        }
    }

    private void stopPythonWorker() {
        if (pythonProcess != null && pythonProcess.isAlive()) {
            pythonProcess.destroy();
        }
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}
        SwingUtilities.invokeLater(() -> new MainFrame().setVisible(true));
    }
}
