import javax.swing.*;
import javax.swing.border.LineBorder;
import javax.swing.plaf.basic.BasicComboBoxUI;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

public class MainFrame extends JFrame {
    private JLabel dropZoneLabel;
    private JTextArea logArea;
    private JTextArea resultInfoArea;
    private JComboBox<String> modeComboBox;

    // Resolve project root directory
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
        throw new IllegalStateException("ACS project root not found. Please set ACS_PROJECT_ROOT.");
    }

    private Process pythonProcess;
    private BufferedWriter pythonInput;
    private BufferedReader pythonOutput;
    private volatile boolean isPythonReady = false;
    private String lastResultImagePath = null;
    private boolean isCameraMode = false;

    public MainFrame() {
        setTitle("ACS - Automatic Candy Selector (Real-time Pipeline)");
        setSize(1180, 800);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        // --- MODERN DARK PALETTE ---
        Color bgDark = new Color(18, 18, 18);
        Color panelDark = new Color(30, 30, 30);
        Color accentOrange = new Color(255, 111, 0);
        Color textLight = new Color(220, 220, 220);
        Color boxDarkBg = new Color(24, 24, 24);

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

        // --- MODE CONTROL PANEL ---
        JPanel modeControlPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 0));
        modeControlPanel.setBackground(panelDark);

        JLabel modeLabel = new JLabel("Select Mode:");
        modeLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        modeLabel.setForeground(textLight);
        modeControlPanel.add(modeLabel);

        String[] modes = {"Image Drop Mode", "Live Camera Mode"};
        modeComboBox = new JComboBox<>(modes);
        modeComboBox.setFont(new Font("Segoe UI", Font.BOLD, 13));
        modeComboBox.setFocusable(false);
        modeComboBox.setCursor(new Cursor(Cursor.HAND_CURSOR));
        modeComboBox.setPreferredSize(new Dimension(190, 34));
        modeComboBox.setBackground(boxDarkBg);
        modeComboBox.setForeground(Color.WHITE);
        modeComboBox.setBorder(new LineBorder(new Color(70, 70, 70), 1, true));

        modeComboBox.setUI(new BasicComboBoxUI() {
            @Override
            protected JButton createArrowButton() {
                JButton arrowBtn = new JButton() {
                    @Override
                    protected void paintComponent(Graphics g) {
                        Graphics2D g2 = (Graphics2D) g.create();
                        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                        g2.setColor(boxDarkBg);
                        g2.fillRect(0, 0, getWidth(), getHeight());
                        g2.setColor(Color.WHITE);
                        int cx = getWidth() / 2 - 2;
                        int cy = getHeight() / 2;
                        int[] xPoints = {cx - 4, cx + 4, cx};
                        int[] yPoints = {cy - 2, cy - 2, cy + 3};
                        g2.fillPolygon(xPoints, yPoints, 3);
                        g2.dispose();
                    }
                };
                arrowBtn.setBorder(BorderFactory.createEmptyBorder());
                arrowBtn.setContentAreaFilled(false);
                return arrowBtn;
            }

            @Override
            public void paintCurrentValueBackground(Graphics g, Rectangle bounds, boolean hasFocus) {
                g.setColor(boxDarkBg);
                g.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
            }
        });

        modeComboBox.setRenderer(new DefaultListCellRenderer() {
            @Override
            public Component getListCellRendererComponent(JList<?> list, Object value, int index, 
                                                          boolean isSelected, boolean cellHasFocus) {
                JLabel renderer = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
                renderer.setOpaque(true);
                renderer.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
                renderer.setFont(new Font("Segoe UI", Font.BOLD, 13));
                renderer.setBackground(isSelected ? accentOrange : boxDarkBg);
                renderer.setForeground(Color.WHITE);
                return renderer;
            }
        });

        modeComboBox.addActionListener(e -> onModeChanged());
        modeControlPanel.add(modeComboBox);
        headerPanel.add(modeControlPanel, BorderLayout.EAST);
        add(headerPanel, BorderLayout.NORTH);

        // --- CENTER REGION (IMAGE VIEW & SUMMARY DETAILS) ---
        JPanel centerPanel = new JPanel(new BorderLayout(15, 0));
        centerPanel.setBackground(bgDark);
        centerPanel.setBorder(BorderFactory.createEmptyBorder(0, 15, 0, 15));

        // 1. Image Viewport & Pure Drag & Drop Zone
        JPanel imageContainerPanel = new JPanel(new BorderLayout());
        imageContainerPanel.setBackground(panelDark);
        imageContainerPanel.setBorder(new LineBorder(new Color(65, 65, 65), 2, true));
        
        dropZoneLabel = new JLabel("<html><div style='text-align: center;'><br><br><b>DRAG & DROP IMAGE HERE</b><br><span style='font-size:12px; color:gray;'>Supports JPG, PNG, WEBP</span></div></html>", JLabel.CENTER);
        dropZoneLabel.setForeground(textLight);
        dropZoneLabel.setFont(new Font("Segoe UI", Font.PLAIN, 16));

        imageContainerPanel.add(dropZoneLabel, BorderLayout.CENTER);
        centerPanel.add(imageContainerPanel, BorderLayout.CENTER);

        // 2. Right: Detection Details Panel
        JPanel infoPanel = new JPanel(new BorderLayout());
        infoPanel.setBackground(panelDark);
        infoPanel.setBorder(BorderFactory.createTitledBorder(
            new LineBorder(new Color(65, 65, 65), 1, true),
            " Detection Summary ", 
            0, 0, new Font("Segoe UI", Font.BOLD, 12), accentOrange
        ));

        resultInfoArea = new JTextArea();
        resultInfoArea.setEditable(false);
        resultInfoArea.setBackground(new Color(24, 24, 24));
        resultInfoArea.setForeground(new Color(240, 240, 240));
        resultInfoArea.setFont(new Font("Segoe UI", Font.PLAIN, 14));
        resultInfoArea.setText("\n  No items processed yet.\n  Please drag & drop an image here\n  or switch to live camera mode.");
        
        JScrollPane infoScroll = new JScrollPane(resultInfoArea);
        infoScroll.setBorder(BorderFactory.createEmptyBorder());
        infoPanel.add(infoScroll, BorderLayout.CENTER);
        infoPanel.setPreferredSize(new Dimension(340, 0));
        centerPanel.add(infoPanel, BorderLayout.EAST);

        add(centerPanel, BorderLayout.CENTER);

        // --- BOTTOM: CONSOLE ACTIVITY LOG ---
        logArea = new JTextArea(5, 40);
        logArea.setEditable(false);
        logArea.setBackground(new Color(12, 12, 12));
        logArea.setForeground(new Color(0, 255, 128));
        logArea.setFont(new Font("Consolas", Font.PLAIN, 13));
        JScrollPane logScroll = new JScrollPane(logArea);
        logScroll.setBorder(BorderFactory.createTitledBorder(
            null, " Activity Log ", 0, 0, new Font("Segoe UI", Font.BOLD, 11), accentOrange
        ));
        add(logScroll, BorderLayout.SOUTH);

        // --- DRAG & DROP LISTENER ---
        new DropTarget(imageContainerPanel, new DropTargetListener() {
            public void dragEnter(DropTargetDragEvent dtde) {}
            public void dragOver(DropTargetDragEvent dtde) {}
            public void dropActionChanged(DropTargetDragEvent dtde) {}
            public void dragExit(DropTargetEvent dte) {}
            
            @SuppressWarnings("unchecked")
            public void drop(DropTargetDropEvent dtde) {
                if (isCameraMode) {
                    appendLog("⚠️ Camera Mode is active. Switch back to Image Drop Mode first.\n");
                    return;
                }
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

        Runtime.getRuntime().addShutdownHook(new Thread(this::stopPythonWorker));
        new Thread(this::initPythonWorker).start();
    }

    private void onModeChanged() {
        int selectedIndex = modeComboBox.getSelectedIndex();
        if (selectedIndex == 1) {
            isCameraMode = true;
            dropZoneLabel.setIcon(null);
            dropZoneLabel.setText("<html><div style='text-align: center;'><br><br><b>LIVE WEBCAM ACTIVE</b><br><span style='font-size:12px; color:gray;'>Press 'q' on the camera window to exit</span></div></html>");
            appendLog(">> Mode switched: [LIVE CAMERA MODE].\n");
            sendCommandToPython("MODE:CAMERA");
        } else {
            isCameraMode = false;
            dropZoneLabel.setIcon(null);
            dropZoneLabel.setText("<html><div style='text-align: center;'><br><br><b>DRAG & DROP IMAGE HERE</b><br><span style='font-size:12px; color:gray;'>Supports JPG, PNG, WEBP</span></div></html>");
            appendLog(">> Mode switched: [IMAGE DROP MODE].\n");
            sendCommandToPython("MODE:FILE");
        }
    }

    private void processSelectedFile(File file) {
        showImageOnDropZone(file.getAbsolutePath());
        if (!isPythonReady) {
            appendLog("⚠️ AI model is still loading into memory. Please wait a moment...\n");
            return;
        }
        sendImageToAI(file.getAbsolutePath());
    }

    private void showImageOnDropZone(String imagePath) {
        ImageIcon icon = new ImageIcon(imagePath);
        Image img = icon.getImage().getScaledInstance(750, 520, Image.SCALE_SMOOTH);
        dropZoneLabel.setIcon(new ImageIcon(img));
        dropZoneLabel.setText("");
    }

    // --- LAUNCH BACKGROUND PYTHON PROCESS AND CONSUME STREAMS ---
    private void initPythonWorker() {
        try {
            appendLog("Connecting to Python backend and loading AI models...\n");

            String python = System.getenv("ACS_PYTHON");
            if (python == null || python.isEmpty()) python = "python";
            ProcessBuilder pb = new ProcessBuilder(python, "-u", "worker.py");
            pb.directory(projectRoot());
            pb.environment().put("PYTHONIOENCODING", "utf-8");
            pb.environment().put("NUMEXPR_MAX_THREADS", "16");
            pb.redirectErrorStream(true);

            pythonProcess = pb.start();
            pythonInput = new BufferedWriter(new OutputStreamWriter(pythonProcess.getOutputStream(), StandardCharsets.UTF_8));
            pythonOutput = new BufferedReader(new InputStreamReader(pythonProcess.getInputStream(), StandardCharsets.UTF_8));

            new Thread(() -> {
                try {
                    String logLine;
                    while ((logLine = pythonOutput.readLine()) != null) {
                        final String str = logLine;

                        // Recognize readiness signal
                        if (str.contains("READY") || str.contains("INFO:Switched") || str.contains("Mode")) {
                            if (!isPythonReady) {
                                isPythonReady = true;
                                appendLog(">>> AI PIPELINE READY! Drop an image or start camera mode.\n\n");
                            }
                        }

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
                            appendLog(str + "\n");
                        }
                    }
                } catch (IOException ignored) {}
            }).start();

        } catch (Exception ex) {
            ex.printStackTrace();
            appendLog("Failed to initialize AI worker: " + ex.getMessage() + "\n");
        }
    }

    private void sendCommandToPython(String command) {
        try {
            if (pythonInput != null) {
                pythonInput.write(command);
                pythonInput.newLine();
                pythonInput.flush();
            }
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void sendImageToAI(String imagePath) {
        try {
            lastResultImagePath = null;
            appendLog(">> Processing image: " + new File(imagePath).getName() + "...\n");
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
            appendLog(">>> DETECTION FINISHED!\n\n");
        }
    }

    private void loadAndDisplayJsonResult(String resultImagePath) {
        try {
            File imgFile = new File(resultImagePath);
            String parentDir = imgFile.getParent();
            String imageName = imgFile.getName();
            
            if (imageName.endsWith("_annote.jpg")) {
                String stem = imageName.substring(0, imageName.length() - 11);
                File jsonFile = new File(parentDir, stem + "_predictions.json");
                
                if (jsonFile.exists()) {
                    String content = new String(Files.readAllBytes(jsonFile.toPath()), StandardCharsets.UTF_8);
                    parseAndShowJsonData(content);
                }
            }
        } catch (Exception e) {
            resultInfoArea.setText("\n  Unable to read detection result JSON.");
        }
    }

    private void parseAndShowJsonData(String jsonContent) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n  DETECTED CANDY BREAKDOWN:\n");
        sb.append("  -----------------------------------\n");

        try {
            if (jsonContent.contains("detections")) {
                int count = 0;
                String[] items = jsonContent.split("\\{\\s*\"id\":");
                for (int i = 1; i < items.length; i++) {
                    String item = items[i];
                    count++;

                    String candyType = extractJsonField(item, "type_predit");
                    String candyColor = extractJsonField(item, "couleur_predite");
                    String confType = extractJsonField(item, "confiance_type");

                    double conf = 0.0;
                    try { conf = Double.parseDouble(confType); } catch(Exception ignored) {}

                    sb.append(String.format("  [%02d] %s / %s\n", count, candyType, candyColor));
                    sb.append(String.format("       Confidence: %.1f%%\n", conf * 100));
                    sb.append("  -----------------------------------\n");
                }
                sb.insert(0, String.format("  Total Detected: %d item(s)\n", count));
            }
        } catch (Exception e) {
            sb.append("  Error parsing JSON payload.");
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
            if (idx == -1) return "N/A";

            int colon = json.indexOf(":", idx);
            int comma = json.indexOf(",", colon);
            int brace = json.indexOf("}", colon);
            
            int end = comma;
            if (comma == -1 || (brace != -1 && brace < comma)) {
                end = brace;
            }
            if (end == -1) return "N/A";

            String val = json.substring(colon + 1, end).trim();
            val = val.replace("\"", "");
            return val;
        } catch (Exception e) {
            return "N/A";
        }
    }

    private void appendLog(String text) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(text);
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
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
