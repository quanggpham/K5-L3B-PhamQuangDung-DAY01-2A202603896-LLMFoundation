# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603896
- **Họ và tên:** Phạm Quang Dũng
- **SEED** (in ở ô đầu notebook): 2239

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh cuối Phần 5. Các hàng 1–3c đo trên toàn bộ tập test (775 tin) ở cột "F1 (test)". Các hàng LLM (4a–4c) chỉ chạy trên tập SUB 40 tin nên cột "F1 (test)" của chúng chính là F1 trên SUB. Độ trễ đo từ văn bản thô đến nhãn, gọi từng tin một.

| Cách tiếp cận | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.831 | 0.710 | 0.249 | 0 |
| 2. BoW + MLP (deep learning) | 0.929 | 0.824 | 0.363 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.901 | 0.919 | 132.701 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.938 | 0.947 | 41.248 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.963 | 0.947 | 45.273 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | 0.872 | 0.872 | 1182.873 | 0.005 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | 0.950 | 0.950 | 1260.737 | 0.053 |
| 4c. LLM nhỏ few-shot (gpt-6-luna) | 0.974 | 0.974 | 1407.991 | 0.019 |

Số đo phụ thuộc CPU/GPU của Colab. Tập SUB chỉ có 40 tin nên một tin sai làm F1 đổi vài điểm.

## 3. Dự đoán và thực tế

**3.1. `hard_eval` — accuracy trên 50 case khó**
- **Đoán:** accuracy tụt mạnh. BoW còn 50–65%, MiniLM 65–80%, LLM 80–92%. Với LLM, nhóm khó nhất là prompt injection.
- **Ra:** BoW 54% (LogReg) và 58% (MLP). MiniLM 68% (kNN), 64% (LogReg), 66% (MLP head). LLM nhỏ 98% (zero-shot) và 100% (prompt phòng thủ). Nhóm prompt injection đúng 100% ở cả hai hàng LLM.
- **Lệch ở đâu:** LLM vượt hẳn khoảng đoán và không dính prompt injection. MiniLM + LogReg (64%) thấp hơn khoảng đoán một chút.
- **Vì sao lệch (theo phần phản ánh của bạn trong notebook):** bạn đã tưởng LLM sẽ dính câu "Ignore all previous instructions". Bạn cho rằng 50 case phân biệt theo nghĩa nên hợp với cách LLM đọc. Notebook không có thí nghiệm riêng để kiểm chứng giả thuyết này.

**3.2. `tradeoff` — đường Pareto 1 − F1 theo độ trễ**
- **Đoán:** trên đường Pareto có BoW+LogReg, BoW+MLP, MiniLM+MLP head và LLM nhỏ few-shot. Bị lấn át là MiniLM+kNN, MiniLM+LogReg và LLM lớn. Cả 3 LLM vượt vạch 50 ms. Nhóm MiniLM "không chắc, có thể sát vạch".
- **Ra:** trên đường Pareto là 1, 2, 3b, 4b, 4c. Bị lấn át là 3a, 3c, 4a. Vượt SLA p95 50 ms là 3a (132.7 ms) và cả 3 LLM. 3b (41.2 ms) và 3c (45.3 ms) đạt SLA.
- **Lệch ở đâu:** trong 7 cách tôi đã nêu tên, đoán sai 3 cách là 3b, 3c và 4b.
- **Vì sao lệch:** trên SUB, 3b và 3c cùng F1 0.947 nhưng 3b nhanh hơn (41.2 so với 45.3 ms) nên 3b lấn át 3c. 4b có F1 SUB 0.950, cao hơn 3b, và không cách nào nhanh hơn mà có F1 ≥ 0.950 nên 4b không bị lấn át. Trên toàn bộ test thì 3c lại hơn 3b (0.963 so với 0.938). Vì SUB chỉ có 40 tin nên khoảng cách nhỏ như vậy nằm trong vùng nhiễu.

**3.3. `emb_tricky` — 6 tin nhắn ngụy trang**
- **Đoán:** mô hình BoW đúng nhiều hơn.
- **Ra:** BoW đúng 4/6, Encoder+LogReg đúng 4/6, hoà.
- **Cụ thể:** BoW sai tin "Exclusive: get a brand new phone at no cost, just reply with your address" còn Encoder+LogReg đúng. Ngược lại, BoW đúng tin "Your mobile number has been chosen for a complimentary holiday..." còn Encoder+LogReg sai. Cả hai cùng sai tin "Last chance! Your account rewards expire tonight...".
- **Vì sao lệch (theo giải thích của bạn trong notebook):** "at no cost" cùng nghĩa với "free" nhưng BoW không nắm được, còn encoder hiểu. Notebook không giải thích vì sao encoder sai tin "complimentary holiday". Chỉ có 6 tin nên kết luận rất yếu.

## 4. Phân tích 50 case khó

**Mức tụt F1** (từ F1 trên SUB sang F1 trên 50 case khó). Hai tập khác thành phần: SUB cân bằng 20 spam/20 ham, còn 50 case có 29 spam và 21 ham. Hàng 4d chưa đo riêng trên SUB, số "trước" lấy của 4a.

| Cách | F1 SUB | F1 50 case khó | Thay đổi |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.710 | 0.378 | −0.332 |
| 2. BoW + MLP | 0.824 | 0.462 | −0.362 |
| 3a. Emb + kNN | 0.919 | 0.714 | −0.205 |
| 3b. Emb + LogReg | 0.947 | 0.640 | −0.307 |
| 3c. Emb + MLP head | 0.947 | 0.653 | −0.294 |
| 4a. LLM nhỏ zero-shot | 0.872 | 0.983 | +0.111 |
| 4d. LLM nhỏ + prompt phòng thủ | (lấy của 4a) 0.872 | 1.000 | +0.128 |

Tụt nhiều nhất là BoW + MLP (−0.362), còn BoW + LogReg có F1 thấp nhất (0.378).

**% đoán đúng theo nhóm case:**

| Nhóm | LogReg | BoW MLP | kNN | 3b | 3c | 4a | 4d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tưởng spam nhưng không phải | 100% | 90% | 70% | 80% | 90% | 90% | 100% |
| Thao túng / lừa đảo | 12% | 50% | 88% | 25% | 25% | 100% | 100% |
| Spam giọng tự nhiên | 20% | 0% | 30% | 40% | 40% | 100% | 100% |
| Né từ khoá | 17% | 17% | 83% | 83% | 83% | 100% | 100% |
| Prompt injection | 67% | 83% | 100% | 100% | 100% | 100% | 100% |
| Nói về spam | 100% | 100% | 25% | 50% | 50% | 100% | 100% |
| Tin giao dịch hợp lệ | 83% | 100% | 83% | 83% | 83% | 100% | 100% |
| **Tổng (50 case)** | **54%** | **58%** | **68%** | **64%** | **66%** | **98%** | **100%** |

**Hai case cụ thể:** BoW + MLP đúng 0% ở nhóm "Spam giọng tự nhiên" (0/10), nên mọi case trong nhóm này đều bị sai, trong đó có:
- #11: "Hey it's Jess from the gym 😊 we're doing a members-only thing this week, 3 months for the price of 1, just reply JOIN and I'll sort it". Bẫy: giọng thân mật, giả làm người quen, thực chất là khuyến mãi.
- #14: "Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back". Bẫy: kể chuyện cá nhân, thực chất là lừa việc làm.

**Cơ chế (theo phân tích của bạn trong notebook):** BoW nhìn token rời rạc, không nắm được ý đồ cả câu. Khi tin ít từ spam điển hình thì điểm không vượt ngưỡng 0.5 nên mô hình đoán ham và bỏ lọt spam. Notebook không in điểm từng tin nên chưa kiểm chứng trực tiếp. Embedding (MiniLM) dựa vào chủ đề và giọng văn nên yếu ở spam giọng tự nhiên, thao túng (3b/3c: 25%) và tin nói về spam (kNN 25%). LLM nhỏ zero-shot chỉ sai 1 case, thuộc nhóm "Tưởng spam nhưng không phải" (90%). Prompt phòng thủ sửa được case đó và không làm hỏng nhóm nào.

## 5. Thử thách cá nhân

**Đề #2 (SEED 2239 mod 6 = 1):** ML — giảm vocab còn 100 từ và tăng lên 5000 từ (`build_vocab(..., max_size)`), train lại, báo F1 test, giải thích xu hướng.

**Cách làm:** với mỗi `max_size` trong {100, 5000}: `build_vocab` trên tập train, `vectorize` train và test, `train_logreg` (lr = 0.5, 300 epoch, tham số mặc định), ngưỡng 0.5, tính `metrics` trên tập test.

| max_size | accuracy | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| 100 | 0.960 | 0.981 | 0.634 | 0.770 |
| 5000 | 0.969 | 0.983 | 0.720 | 0.831 |

Tham chiếu từ Phần 1 (vocab 2000, cùng lr và epoch): F1 0.831, precision 0.983, recall 0.72.

**Giải thích (của bạn):** tăng vocab từ 100 lên 5000 thì F1 tăng từ 0.770 lên 0.831, precision gần như không đổi, phần tăng chủ yếu là recall. Vocab 100 từ chỉ giữ các từ phổ biến nhất, phần lớn là từ chung chung, còn các từ khoá spam hiếm hơn không nằm trong vocab nên mô hình bỏ lọt nhiều spam. Precision không đổi vì những từ spam rõ nhất ("call", "free", "txt") đã có sẵn trong vocab nhỏ. Lợi ích giảm dần: từ càng hiếm thì càng ít tin chứa nó, mô hình khó học được trọng số đáng tin và dễ overfit. Từ 2000 lên 5000 từ, F1 không đổi (0.831), phù hợp với nhận xét này. BoW không hiểu nghĩa nên tăng vocab cũng không cứu được tin viết né từ khoá. Lưu ý: tập test chỉ có 82 spam nên chênh recall khoảng 7 tin; xu hướng hợp lý nhưng chưa phải kết luận chắc chắn.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** p95 < 50 ms/tin cho 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây).

**Loại bỏ:**
- Cả 3 cách LLM đều có p95 trên 1 giây (1183 ms, 1261 ms, 1408 ms), hơn 20 lần ngưỡng 50 ms.
- 3a (MiniLM + kNN) có p95 132.7 ms, cũng vượt SLA.

**Đạt SLA:** BoW + LogReg (0.249 ms), BoW + MLP (0.363 ms), 3b (41.2 ms), 3c (45.3 ms).

**Chọn:** Encoder (MiniLM) + MLP head (3c). Lý do: F1 test cao nhất trong nhóm đạt SLA (0.963 so với 0.938 của 3b và 0.929 của BoW + MLP) và không tốn phí API. Phương án dự phòng là 3b (p95 41.2 ms, F1 test 0.938) nếu cần dư địa độ trễ.

**Đánh đổi chấp nhận:**
- Lề so với SLA mỏng: p95 của 3c là 45.3 ms so với ngưỡng 50 ms, và số đo phụ thuộc phần cứng của Colab.
- Thông lượng đo được của 3c là khoảng 26 tin/s (chạy cả lô). Mức tải trung bình khoảng 116 tin/s, nên một bản triển khai như đã đo chưa đủ tải; cần nhiều bản sao hoặc phần cứng mạnh hơn (đây là phép tính từ số đo, chưa được thử nghiệm).
- Chất lượng trên 50 case khó của 3c thấp: F1 0.653, accuracy 66%, yếu ở spam giọng tự nhiên, thao túng và tin nói về spam. LLM nhỏ đạt 98–100% ở bộ này nhưng không đạt SLA.
- Chi phí nếu dùng LLM nhỏ (gpt-6-luna, 100 token input + 1 token output mỗi tin, giá $0.10 / $0.50 mỗi 1M token): khoảng $10.5 cho 1 triệu tin, tức khoảng $105/ngày và $3,150/tháng cho 10 triệu tin/ngày. Model lớn (gpt-6.1-terra) đắt gấp 10 lần theo bảng giá, khoảng $31,500/tháng.

**Hướng mở rộng (đề xuất, chưa đo trong notebook):** mô hình lai — 3c xử lý mọi tin; chỉ tin lưỡng lự (p trong khoảng 0.3–0.7) chuyển cho LLM nhỏ, kèm chuẩn hoá đầu vào và lọc prompt injection trước LLM. Cách này giữ phần lớn tin trong SLA và dùng LLM cho ca khó, nhưng tỉ lệ tin lưỡng lự, chi phí và p95 thực tế của tin đó chưa được đo; tin lưỡng lự đi qua LLM sẽ chậm hơn nên cần đặt timeout.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| 0.1 Data leakage | Tin trùng nằm ở cả train và test thì điểm test bị ảnh hưởng thế nào, vì sao gọi là leakage? | Giải thích khái niệm. | Tự tổng hợp lại câu trả lời 0.1. |
| 1.1 Vector hoá | BoW là gì, giải thích đề, đi từng bước vì chưa quen cú pháp. | Giải thích khái niệm và cú pháp từng bước, đưa khung hàm có chỗ trống. | Tự code; AI chỉ ra lỗi `x` thay vì `text`, nên dùng `tokenize`; chạy test. |
| 1.2 Sigmoid / predict_proba | Giải thích khái niệm, rồi hỏi phương hướng viết code, cú pháp. | Giải thích logit/sigmoid và đưa ra phương hướng coding. | Chạy test, thử `sigmoid(0)` ra 0.5. |
| 1.3 Gradient descent | Giải thích epoch, lr, loss, w và b ở đâu ra. | Giải thích bằng ví dụ và đưa khung vòng lặp 4 bước. | Chép theo khung, chạy, xem loss giảm từ 0.693. |
| 1.4 Metrics | Giải thích precision/recall/F1 và hướng viết hàm, cú pháp python. | Giải thích TP/FP/FN, đưa ra phương hướng; chỉ ra lỗi return set thay vì dict. | Sửa dòng return, chạy test. |
| Đọc lỗi, thí nghiệm 1.B, Bắt lỗi 1 | Giải thích dòng cập nhật w với từ "free", hiện tượng lr=50, 2 lỗi phương pháp. | Giải thích và soạn nháp câu trả lời. | Đối chiếu đồ thị thật, sửa lại lời văn. |
| 2.1–2.2 MLP | Giải thích MLP, ReLU, cách đếm tham số, 5 dòng vòng lặp train. | Giải thích; gợi ý phương hướng code và cú pháp, method; tính 128 129 tham số. | Coding, chạy test, kiểm tra bằng `sum(p.numel())`, đọc đồ thị overfit. |
| 2.A, 2.B | Vì sao MLP không thắng rõ, bỏ ReLU thì sao. | Giải thích; chỉnh lại dự đoán (logistic regression chứ không phải linear regression). | Chạy lại, so F1. |
| 3.1–3.3 Cosine, top-k, kNN | Giải thích khái niệm, nhờ gợi ý phương hướng coding. | Đưa ra phương hướng, gợi ý method, cú pháp cho 3 hàm; và giải thích. | Coding, chạy test đến khi qua. |
| Câu hỏi 3, Bắt lỗi 2 | Đối chiếu 3.B, vì sao spam/ham tách vùng, 3b so với 3c. | Gợi ý kiến thức liên quan. | Viết lại câu trả lời. |
| 4. LLM API | `call_openai`, prompt, `parse_label`, `llm_classify_batch`; vì sao F1 = 0. Nhờ giải thích các method thường dùng trong việc call API. | Viết code; giải thích. | Tự chạy thử. |
| Pareto, stress test, tính tay, phản ánh cuối | Dự đoán, đọc bảng kết quả, cách tính, soạn câu trả lời. | Hướng dẫn tính. | Thực hành tính toán rồi trình bày vào giấy. |
| Thử thách cá nhân #2 | Nhờ giải thích các khái niệm trong đề bài. | Giải thích khái niệm xuất hiện trong đề bài. | Coding bài tập. |

**Điều đã học (của bạn):** mình đoán sai nhiều nhất ở LLM: tưởng nó sẽ dính prompt injection và chỉ đạt 80–92% trên 50 case khó, thực tế nó đạt 98%, còn embedding chỉ 64–68%. Điểm cao trên tập test cùng phân phối không nói lên model có hiểu thật hay không. Đổi cách biểu diễn (từ đếm từ sang hiểu nghĩa) quan trọng hơn đổi độ phức tạp của mô hình. Dự đoán chỉ có giá trị khi đem đối chiếu với số thật.
