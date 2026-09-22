# GH-LBM CUDA Kernel 手寫教學

> 版本日期：2026-09-22 ｜ 匯出自 Claude Docs（rev 52）

從現有的 taichi D2Q9 出發，七個里程碑走到 D3Q19 MRT 的手寫 CUDA kernel。這是**與主線並行的學習軌**，目的是讓你有能力判斷要不要換掉 taichi，而不是預設一定要換。主計畫在文件 **01**，實作細節在 **02**，理論背景在 **03**。

## 這份文件的定位

這是一條**與主線並行的學習軌**。主線是 Rhino Grasshopper 即時風環境 LBM 工具，引擎目前用 taichi，而且它能跑。這份教學存在的理由不是「準備換掉 taichi」，是**讓你有能力自己判斷該不該換**。

判斷需要證據，證據需要你能自己寫出一個對照組。沒寫過 CUDA 的人沒辦法判斷 taichi 產生的 code 是好是壞——你只能相信它。寫過之後，你會知道理論上限在哪，也就知道 taichi 離上限多遠。

走完七個里程碑，三種結局都是成功：

| 結局 | 什麼情況下成立 |
| --- | --- |
| **留著 taichi** | 你手寫的 CUDA 跟 taichi 版差距在 20% 以內。那 20% 換不來 taichi 的開發速度、跨平台、跟 Python 生態的黏合度。 |
| **只換最熱的 kernel** | collision + streaming 那個融合 kernel 佔 90% 時間、手寫快 2 倍以上，其餘（邊界、後處理、VTK 輸出）留在 taichi。 |
| **整包換成 CUDA** | 你要的功能（FP16 儲存、Esoteric Pull、multi-GPU domain decomposition）taichi 做不到或做起來比手寫還痛苦，而且效能差距 3 倍以上。 |

第三種結局最少見，但**它正好是 L6 的內容會決定的**。所以這條路值得走完。

---

## 先講結論：LBM 是 memory-bandwidth bound

在你寫任何一行 CUDA 之前，先把這件事刻進腦袋裡，因為它會推翻你大部分關於「最佳化」的直覺。

Lattice Boltzmann Method（LBM，格子波茲曼法）每一格每一步做的事：

1. 從記憶體讀出 19 個分布函數（distribution function，DDF）
2. 算出密度、速度、碰撞（collision）
3. 把 19 個值寫回記憶體（streaming，遷移）

算術部分即使用 MRT（Multiple-Relaxation-Time，多鬆弛時間）這種「昂貴」的碰撞模型，也不過是兩次 19×19 矩陣乘法，約 722 次乘加。記憶體部分：

```latex
\text{雙陣列 f32：} \quad (19_{\text{read}} + 19_{\text{write}}) \times 4\,\text{bytes} = 152\ \text{bytes / cell / step}
```

把兩者放在一起看，用 RTX 4090 的規格（約 1008 GB/s、f32 算力約 80 TFLOP/s）：

```latex
\text{記憶體時間} = \frac{152\ \text{B}}{1008 \times 10^{9}\ \text{B/s}} = 1.51 \times 10^{-10}\ \text{s / cell}
```

```latex
\text{算術時間} = \frac{722 \times 2\ \text{FLOP}}{80 \times 10^{12}\ \text{FLOP/s}} = 1.81 \times 10^{-11}\ \text{s / cell}
```

記憶體慢了 **8 倍**。也就是說：即使你把 MRT 碰撞完全免費化（改成 BGK、或用 tensor core），總時間最多只省 12%。反過來，只要把每格搬的 bytes 減半，速度就直接接近兩倍。

這推出三個判斷原則，貫穿整份文件：

1. **記憶體佈局 > 演算法最佳化 > 語言選擇。** 換語言（taichi → CUDA）通常只影響第三順位。
2. **所有效能數字都用 MLUPS 表示，並且永遠跟「理論上限」並列。** 不並列的效能數字沒有意義。
3. **「快了 30%」不是結論，「達到頻寬上限的 85%」才是結論。** 後者告訴你還剩多少空間；前者什麼都沒告訴你。

### 那 FP16 + in-place 能省多少

主線文件（01/02）給的目標已經修正成一道階梯：repo 現況（四個 kernel、三次全陣列往返）約 488 B/cell/step → 融合 collide-stream 後 152 B（3.2×）→ 加上 FP16 儲存 76 B（累計 6.4×）→ in-place 流量不變仍是 76 B，但記憶體佔用降到 38 B/cell。最大的單一收益是 kernel 融合；FP16 是流量槓桿、in-place 是容量槓桿，兩者不相乘。我把帳攤開來講，因為這筆帳怎麼算需要說清楚：

| 方案 | 每 cell 每 step 的 DRAM 流量 | 記憶體佔用（每 cell） |
| --- | --- | --- |
| 雙陣列 f32 | 19×4 讀 + 19×4 寫 = **152 B** | 38×4 = 152 B |
| 雙陣列 f16 | 19×2 + 19×2 = **76 B** | 76 B |
| 單陣列（in-place）f32 | 19×4 + 19×4 = **152 B** | 19×4 = **76 B** |
| 單陣列 f16 | 19×2 + 19×2 = **76 B** | 19×2 = **38 B** |

**誠實標註**：38 bytes 這個數字對應的是「單陣列 FP16 的記憶體**佔用**」，或是「19 個值 × 2 bytes 只算一次」。如果讀跟寫分開計，單陣列 FP16 的**流量**是 76 B，相對於 f32 雙陣列是 **2 倍**，不是 4 倍。

那 in-place 到底省什麼？省的是：

- **記憶體佔用減半** → 同一張卡能跑兩倍大的域。這對風環境模擬（你需要整個街廓）是直接的功能性提升，不只是速度。
- **write-allocate 的浪費**。雙陣列寫入目標陣列時，若沒寫滿整條 cache line，硬體要先把那條 line 從 DRAM 讀進來（read-for-ownership）。in-place 時讀寫命中同一條 line，這筆讀取不存在。實際省多少取決於你的 coalescing 做得好不好——做得完美時省 0，做得差時省很多。

所以：**保守估 2 倍，樂觀可能更多，L7 就是去量出真實值的地方。** 不要在量測前就把 4 倍寫進提案。

---

## 總時程

假設每週 15–20 小時的研究節奏：

| 里程碑 | 內容 | 估計 | 累計 |
| --- | --- | --- | --- |
| L1 | CUDA 基礎、vector add | 1 週 | 1 |
| L2 | D2Q9 手寫 CUDA，逐行移植 | 2 週 | 3 |
| L3 | Coalescing：AoS vs SoA 實測 | 1 週 | 4 |
| L4 | D2Q9 in-place（AA-pattern） | 1.5 週 | 5.5 |
| L5 | D3Q19 + MRT | 3 週 | 8.5 |
| L6 | FP16 儲存 + Esoteric Pull | 2 週 | 10.5 |
| L7 | 完整 benchmark 與決策 | 1 週 | 11.5 |

**約三個月。** L5 是最陡的一段，L2 是最容易卡住的一段（因為第一次要跟 host code、編譯、Python 綁定全部同時打交道）。如果時間緊，可跳過 L4 直接到 L5，但**不要跳過 L3**——L3 是整條軌道 ROI 最高的一週。

---

## L1 — CUDA 基礎：執行模型、記憶體層次、vector add

**時間估計：1 週（15–20 小時）**

### 學習目標

- 能解釋 grid / block / thread 三層結構，並手算任意 thread 的全域索引
- 知道 warp 是什麼、為什麼是 32、divergence 為什麼是問題
- 能說出 global / shared / register / constant / local 各自的延遲與容量量級
- 寫出、編譯、執行一個有正確錯誤檢查的 vector add
- 能從 Python 呼叫自己寫的 kernel（至少一種方式）

### 關鍵觀念

#### 三層執行結構

taichi 幫你藏起來的東西在這裡要自己管。`ti.kernel` 裡的 `for i, j in x:` 在 taichi 是一行，在 CUDA 是這樣拆的：

```
grid                        ← 一次 kernel launch 的全部
 └─ block (thread block)    ← 排程的單位，跑在同一個 SM 上
     └─ warp (32 threads)   ← 硬體真正的執行單位
         └─ thread          ← 你寫的程式碼的視角
```

三個關鍵事實，順序就是重要性順序：

1. **同一個 block 內的 thread 可以透過 shared memory 溝通、可以用 `__syncthreads()` 對齊。** 不同 block 之間**不行**（在一次 launch 內沒有可靠的同步機制）。
2. **block 的大小你選，grid 的大小你算。** block 通常選 128 或 256（必須是 32 的倍數，上限 1024）。
3. **block 的執行順序完全不保證。** 不要寫任何依賴 block 順序的程式。

全域索引的演算法，1D 版本：

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

`blockDim.x` 是「每個 block 有幾個 thread」，`blockIdx.x` 是「我是第幾個 block」，`threadIdx.x` 是「我在 block 裡的第幾個」。2D / 3D 就是多幾個維度：

```cuda
// 2D，對應 taichi 的 for i, j in field:
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i >= nx || j >= ny) return;   // 邊界守衛，幾乎永遠需要
int idx = j * nx + i;             // 注意：i 放在連續維度（見 L3）
```

那個 `if (i >= nx) return;` 不是可選的。grid 大小要用**無條件進位除法**算，所以最後一個 block 幾乎一定有多餘的 thread：

```cuda
dim3 block(32, 8);                                    // 256 threads
dim3 grid((nx + block.x - 1) / block.x,
          (ny + block.y - 1) / block.y);
my_kernel<<<grid, block>>>(args...);
```

#### warp 是什麼，為什麼是 32

硬體不是一次排程一個 thread，是一次排程 **32 個** thread 組成的 warp。同一個 warp 裡的 32 個 thread **在同一個時鐘週期執行同一條指令**（SIMT，Single Instruction Multiple Thread）。

這帶來兩個後果：

**（1）Warp divergence（分支發散）**：如果同一個 warp 裡有些 thread 走 `if`、有些走 `else`，硬體會**兩條路都走一遍**，不走的那些 thread 被遮蔽掉。兩條路都不便宜的話，這個 warp 的執行時間就是兩條路的總和。

對 LBM 的意義：`if (solid[idx] == 0) { ... }` 這種固體節點判斷，如果固體分布很散亂（多孔介質），幾乎每個 warp 都會發散。如果固體是整塊的（建築量體），大部分 warp 落在純流體或純固體區，發散很少。**你的風環境案例屬於後者，這是好消息。**

**（2）Coalescing（記憶體合併）**：一個 warp 的 32 個 thread 同時發出的記憶體請求，硬體會嘗試合併成最少的交易。這是 L3 的主題，也是整份文件最重要的一節。

為什麼是 32？這是 NVIDIA 的硬體設計選擇，從 G80（2006）到現在所有架構都是 32，沒有變過。程式裡可以用內建變數 `warpSize`，但實務上大家直接寫 32。AMD 的對應物叫 wavefront，是 64（RDNA 之後可選 32）。

#### 記憶體層次

這張表是你接下來三個月會不斷回來查的東西。數字是 **Ampere / Ada 世代的量級**，不同架構會變：

| 層級 | 延遲（cycles） | 容量 | 範圍 | 誰控制 |
| --- | --- | --- | --- | --- |
| Register | ~1 | 每 SM 64K 個 32-bit reg；每 thread 上限 255 | 單一 thread | 編譯器 |
| Shared memory | ~20–30 | 每 block 預設 48 KB，可 opt-in 更多 | 一個 block | **你** |
| L1 cache | ~30 | 與 shared 共用，每 SM 約 128 KB | 一個 SM | 硬體 |
| Constant cache | ~5（命中且 warp 內同址廣播） | 總計 64 KB，每 SM 約 8 KB cache | 全 grid 唯讀 | 你（宣告 `__constant__`） |
| L2 cache | ~200 | 數 MB 到數十 MB（GA102 約 6 MB、AD102 約 72 MB、A100 40 MB） | 全 GPU | 硬體 |
| Global memory (DRAM) | **~400–800** | 8–80 GB | 全 GPU + host | 你（`cudaMalloc`） |
| Local memory | 同 global（~400–800） | — | 單一 thread | 編譯器（register spill 時） |

**要記住的量級關係：register 比 global memory 快約 500 倍。** 這也是為什麼「把 19 個 DDF 讀進 register、全部算完、一次寫回」是 LBM kernel 的標準寫法。

**Local memory 是陷阱。** 名字聽起來像快的，其實它就是 global memory 的一塊私有區域，延遲一樣慘。當你的 kernel 用太多 register（D3Q19 MRT 很容易），編譯器會把放不下的變數 spill 到 local memory，效能斷崖式下跌。用 `nvcc -Xptxas -v` 可以看到 register 用量和 spill 量：

## L2 — D2Q9 手寫 CUDA：從現有 taichi 版逐行移植

**時間估計：2 週（30–40 小時）**

這是整條軌道最容易卡住的一段，不是因為難，是因為第一次要同時面對：kernel 邏輯、記憶體管理、host code、編譯、Python 綁定、數值驗證。**策略是把驗證放在最前面**——先建立「怎麼知道對不對」的機制，再寫 kernel。

### 學習目標

- 把一個能跑的 taichi D2Q9 逐行翻成 CUDA，結果數值上可驗證地一致
- 建立起一套「taichi 版 vs CUDA 版」的自動比對流程，之後每個里程碑都會重用
- 理解 LBM 資料佈局的第一個決策點

### 關鍵觀念

#### D2Q9 速度集

```latex
\mathbf{c}_0 = (0,0),\quad
\mathbf{c}_{1..4} = (\pm 1, 0), (0, \pm 1),\quad
\mathbf{c}_{5..8} = (\pm 1, \pm 1)
```

```latex
w_0 = \frac{4}{9},\quad w_{1..4} = \frac{1}{9},\quad w_{5..8} = \frac{1}{36},\quad c_s^2 = \frac{1}{3}
```

**重要：照抄你 taichi 版的方向編號順序，不要用教科書的順序。** 這一步的唯一目的是逐格比對，編號一旦不同，比對就得先做排列對應，白白多一個出錯來源。

順帶一提，你可以從 repo 的 D3Q19 看到同一件事怎麼做：`Single_phase/LBM_3D_SinglePhase_Solver.py` 裡的 `e_f[0..18]` 定義了那份程式碼自己的方向順序，而 `self.LR = [0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17]` 是對應的「反向方向」查表。這兩樣東西在 CUDA 版要原封不動照搬。

#### 資料佈局的第一個決策

D2Q9 每格 9 個 float。兩種擺法：

```cuda
// AoS (Array of Structs)：同一格的 9 個值連續
//  記憶體： [cell0.f0 .. cell0.f8][cell1.f0 .. cell1.f8] ...
#define IDX_AOS(x, y, q, nx, ny)  (((y) * (nx) + (x)) * 9 + (q))

// SoA (Struct of Arrays)：同一方向的所有格連續
//  記憶體： [f0 of all cells][f1 of all cells] ...
#define IDX_SOA(x, y, q, nx, ny)  ((q) * (nx) * (ny) + (y) * (nx) + (x))
```

**L2 用 SoA。** 理由在 L3 會完整量給你看，現在先接受這個結論並照做——因為若 L2 用 AoS 寫完，L3 要改的時候整個 kernel 都要動。

注意 SoA 裡 `x` 是最連續的維度：相鄰 thread 的 `x` 相差 1，位址相差 4 bytes。這是 coalescing 的前提。

**關於 taichi 這邊**：`ti.Vector.field(9, ti.f32, shape=(nx, ny))` 預設是哪種佈局？taichi 文件裡「Fields (advanced)」一節談 SNode 與 AoS/SoA，可以用 `ti.root.dense(...).place(...)` 明確指定。**我不確定各版本的預設值，這正是 L1 附節那個 `print_ir=True` 的第一個實際用途——自己印出來看。** 不要憑記憶，去看 IR。

#### Kernel 該怎麼切

**一個 thread 負責一個 cell。** 這是 LBM 在 GPU 上的標準做法，幾乎沒有例外。原因：

- 每個 cell 的工作量固定（9 或 19 個 DDF），負載天然平衡
- 每個 cell 的 register 需求剛好塞得下（D2Q9 約 20–30 個 register）
- 不需要 shared memory——這點違反直覺，但**對 LBM 來說 shared memory 通常不值得**。streaming 的鄰居存取在 L1/L2 cache 已經被吃掉了，手動搬進 shared memory 反而增加指令數和同步成本。文獻上有用 shared memory 做 streaming 的作法，在較舊的架構上有效；Ampere 之後普遍認為直接靠 cache 更好。**這是可以在 L3 之後自己實測驗證的假設。**

#### 兩個 kernel 還是一個

taichi 版本大概是分開的（repo 的 3D 版就是 `colission()` / `streaming1()` / `Boundary_condition()` / `streaming3()` 四個 kernel）。CUDA 版**應該融合成一個**：

```
分開：collide kernel 讀 f 寫 f → stream kernel 讀 f 寫 F
      流量 = 4 × 9 × 4 bytes = 144 B/cell/step

融合：一個 kernel 讀 f（含鄰居）、算、寫 F
      流量 = 2 × 9 × 4 bytes = 72 B/cell/step
```

**融合直接省一半頻寬。** 這是你的 CUDA 版對上 taichi 版的第一個、也可能是最大的結構性優勢——不是因為 CUDA 比較快，是因為你換了演算法結構。

（公平起見：taichi 也做得到融合，把 collision 跟 streaming 寫在同一個 `@ti.kernel` 裡即可。所以這筆帳嚴格講不該算在「CUDA vs taichi」上。**L7 做比較時要記得把 taichi 版也融合過再比，否則就是不公平的比較，會得到錯誤的決策。**）

#### Pull 還是 push

融合 kernel 有兩種寫法：

- **Push（推）**：讀自己的 f，碰撞，把結果**寫到鄰居**。寫入是分散的。
- **Pull（拉）**：**從鄰居讀**上一步的 f，碰撞，寫回自己。讀取是分散的。

**選 Pull。** 理由：GPU 上分散的讀比分散的寫便宜（讀可以靠 cache 和 latency hiding 吸收，寫有 write-allocate 成本）；而且 Pull 讓每個 thread 只寫自己的格子，不需要原子操作、不會有寫入衝突。

## L3 — 記憶體合併（Memory Coalescing）：AoS vs SoA 實測

**時間估計：1 週（15–20 小時）**

**這是整條軌道 ROI 最高的一週。** 不要跳過，也不要只讀不做——這一節的價值在於你親手量出那個倍數，而不是讀到那個倍數。

### 學習目標

- 能解釋 coalescing 的硬體機制，精確到 sector 層級
- 設計並執行一個乾淨的 AoS vs SoA 對照實驗
- 會算 MLUPS，會從頻寬反推理論上限
- 能用 Nsight Compute 讀出「每個請求用了幾個 sector」

### 關鍵觀念

#### Coalescing 的機制

GPU 的 DRAM 不是按 byte 存取的，是按 **sector** 存取。在 Ampere / Ada 上，L1 到 L2 的傳輸單位是 **32-byte sector**（cache line 是 128 bytes = 4 sectors）。

當一個 warp 的 32 個 thread 同時發出載入請求，硬體會看這 32 個位址落在幾個 sector 裡，然後為每個 sector 發一次交易。

**情況 A：完美合併（SoA）**

thread 0..31 讀 `src[q*N + cell]`，`cell` 連續，每個 4 bytes：

```
位址：  base+0  base+4  base+8  ...  base+124
sector: [--0--][--1--][--2--][--3--]     ← 32 threads × 4 B = 128 B = 4 sectors
```

請求 128 bytes，傳輸 128 bytes。**效率 100%，每個請求 4 個 sector。**

**情況 B：跨步存取（AoS，D3Q19）**

thread 0..31 讀 `src[cell*19 + q]`，相鄰 thread 位址相差 19×4 = 76 bytes：

```
位址：  base+0   base+76   base+152  ...  base+2356
sector: [-0-]      [-2-]     [-4-]   ...      ← 每個 thread 幾乎落在不同 sector
```

76 > 32，所以幾乎每個 thread 獨佔一個 sector。32 個請求 → **32 個 sector 交易 = 1024 bytes 傳輸，但只用到 128 bytes。**

```latex
\text{效率} = \frac{32 \times 4}{32 \times 32} = \frac{128}{1024} = 12.5\%
```

**浪費 8 倍頻寬。** 而 LBM 是純 bandwidth bound，所以這 8 倍會直接反映在 MLUPS 上（實務上不會到滿 8 倍，因為 L2 cache 會吃掉一部分重複——鄰近 thread 讀的 sector 之後會被其他 q 用到。實測通常在 **2–5 倍**之間，取決於 cache 大小和域的大小。**這就是你要量的數字。**）

**情況 C：AoS + `float4` 向量化載入**

AoS 也不是完全沒救。如果每個 thread 一次載入 16 bytes（`float4`），32 個 thread 就是 512 bytes，需要 16 個 sector，效率 100%（但一次只拿到 4 個 q）。D3Q19 的 19 不是 4 的倍數，補成 20 或 24 可以用這招。**不過這樣做的複雜度跟收益比不上直接用 SoA，通常只在被迫用 AoS 的情境才考慮。**

#### 為什麼 taichi 的預設可能不理想

`ti.Vector.field(19, ti.f32, shape=(nx,ny,nz))`——這個宣告裡的 `19` 是向量分量，taichi 要決定把它放在記憶體的哪一層。如果它擺在最內層（AoS），你就吃到情況 B。

repo 的 `Single_phase/LBM_3D_SinglePhase_Solver.py` 第 32–33 行正是這樣宣告的：

```python
self.f = ti.Vector.field(19, ti.f32, shape=(nx,ny,nz))
self.F = ti.Vector.field(19, ti.f32, shape=(nx,ny,nz))
```

**這是本週最有價值的一個調查題：taichi 實際把它排成什麼樣子？** 三個查法：

1. `ti.init(arch=ti.cuda, print_ir=True)`，看 IR 裡的位址計算
2. 用 `ti.root.dense(ti.ijk, (nx,ny,nz)).place(f.get_scalar_field(0), ..., f.get_scalar_field(18))` 明確指定 SoA，跟預設版本比 MLUPS
3. 直接用 Nsight Compute 量 taichi 產生的 kernel 的 `sectors per request`

**我不確定 taichi 各版本對 vector field 的預設佈局，所以不要引用我的說法，去查 taichi 官方文件的「Fields (advanced)」與「Layout」章節，並用上面三個方法之一實證。** 如果結果是「taichi 預設就是 SoA 且效率不錯」，那本身就是「留著 taichi」這個結局的一個有力證據。

#### MLUPS 的定義與計算

**MLUPS = Million Lattice Updates Per Second**（百萬格點更新每秒）：

```latex
\text{MLUPS} = \frac{n_x \times n_y \times n_z \times N_{\text{steps}}}{t_{\text{seconds}} \times 10^{6}}
```

這是 LBM 界的通用單位，所有論文都用它，所以你的數字可以直接跟文獻比。

**幾個使用規則**：

- 分子用**總格點數**，包含固體格點（除非你用 sparse storage，那要另外說明）
- `t` 只算迴圈時間，不含初始化、不含 VTK 輸出
- 一定要暖機後再計時（L7 會詳談）
- 報數字時一定附上：GPU 型號、域大小、精度、碰撞模型（BGK/MRT）、是否雙陣列

#### 從頻寬反推理論上限

```latex
\text{MLUPS}_{\max} = \frac{BW_{\text{effective}}\ [\text{B/s}]}{B_{\text{per cell per step}}\ [\text{B}]} \times 10^{-6}
```

用 `bandwidthTest` 量到的 Device-to-Device 值當 `BW_effective`（不要用規格書的理論值，那永遠達不到）。

**幾張常見卡的數字**（規格頻寬，實測約 80–90%）：

## L4 — D2Q9 in-place streaming（AA-pattern）

**時間估計：1.5 週（20–30 小時）**

### 學習目標

- 理解雙陣列為什麼在記憶體佔用上浪費一半
- 實作 AA-pattern，結果與雙陣列版逐格一致
- 量出記憶體佔用減半、並誠實量出速度變化（可能不大）

### 關鍵觀念

#### 雙陣列浪費的到底是什麼

先把話說清楚，因為這裡很容易誤解：

**雙陣列浪費的是記憶體「佔用」，不是必然浪費「頻寬」。**

一個融合的 pull kernel，雙陣列版每步是：19 讀 + 19 寫 = 38 次存取。in-place 版也是：19 讀 + 19 寫 = 38 次存取。**流量一樣。**

真正省下的是：

1. **記憶體佔用減半。** D3Q19 f32，256³ 的域：

```latex
\text{雙陣列: } 256^3 \times 38 \times 4\ \text{B} = 2.55\ \text{GB}
```

```latex
\text{單陣列: } 256^3 \times 19 \times 4\ \text{B} = 1.27\ \text{GB}
```

對 12 GB 的卡，這是「能不能跑 384³」和「只能跑 256³」的差別。**對你的風環境案例，域的大小就是能不能涵蓋整個街廓，這是功能性的，不只是效能。**

1. **Write-allocate 的消除。** 寫入目標陣列時，若一條 cache line 沒被整條寫滿，硬體必須先從 DRAM 把它讀上來。SoA + 完美 coalescing 時這件事幾乎不發生；但只要你的 nx 沒對齊、或 block size 導致邊緣沒寫滿，就會發生。in-place 時讀和寫命中同一條 line，這筆成本天然不存在。
2. **L2 cache 的有效容量加倍。** 工作集小了一半，同樣大小的 L2 能涵蓋更大的域。這在中等大小的域上會有可觀的間接效益。

**誠實的期待值：速度提升大概在 0–30% 之間，視域大小和對齊而定。記憶體減半則是確定的 2 倍。** 不要期待 AA-pattern 本身帶來兩倍速度——那不是它的機制。它真正的價值是「解鎖更大的域」以及「為 L6 的 FP16 鋪路」（FP16 的 2 倍才是速度的來源）。

#### AA-pattern 的機制

Bailey 等人 2009 年提出。核心想法：**streaming 不需要真的搬資料，只需要改變「從哪讀、往哪寫」的對應關係**。用奇偶兩種步驟交替，讓資料在同一塊記憶體裡「原地翻身」。

規則（設 `q̄` 為 `q` 的反向）：

**偶數步（even / "A" step）——完全本地，不碰鄰居：**

```
讀：f[q]      ← 自己格子的 slot q
寫：slot q̄    ← 自己格子的反向 slot
```

**奇數步（odd / "B" step）——碰鄰居，讀寫同一個位置：**

```
讀：slot q    ← 鄰居（沿 -c_q 方向）的 slot q
寫：slot q̄    ← 同一個鄰居的 slot q̄
```

為什麼這樣就對？直覺是：偶數步把「應該要往外送的值」存在反向的 slot 裡（等於把它「預先轉向」），奇數步再去鄰居那裡把它取回並放回原位。兩步合起來等於兩次完整的 collide+stream。

```cuda
// AA-pattern，D2Q9，SoA。注意：src 和 dst 是同一塊記憶體，所以沒有 __restrict__
__global__ void lbm_aa(float* f, const uint8_t* solid,
                       int nx, int ny, float omega, int even)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int cell = y * nx + x;
    if (solid[cell]) return;

    const int N = nx * ny;
    float fq[9];
    int   addr[9];          // 記下每個 q 的「寫回位址」

    // ---- 讀 ----
    if (even) {
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            fq[q]   = f[q * N + cell];        // 本地讀 slot q
            addr[q] = OPP[q] * N + cell;      // 本地寫 slot q̄
        }
    } else {
        #pragma unroll
        for (int q = 0; q < 9; ++q) {
            int xs = x - CX[q], ys = y - CY[q];
            int nb;
            if (xs < 0 || xs >= nx || ys < 0 || ys >= ny || solid[ys*nx+xs]) {
                nb = cell;                     // bounce-back：退回自己
                fq[q]   = f[OPP[q] * N + nb];
                addr[q] = q * N + nb;
            } else {
                nb = ys * nx + xs;
                fq[q]   = f[q      * N + nb];
                addr[q] = OPP[q]   * N + nb;
            }
        }
    }

    // ---- 碰撞（跟雙陣列版完全一樣的程式碼）----
    collide_bgk(fq, omega);

    // ---- 寫 ----
    #pragma unroll
    for (int q = 0; q < 9; ++q) f[addr[q]] = fq[q];
}
```

Host 端：

```cuda
for (int step = 0; step < nsteps; ++step) {
    lbm_aa<<<grid, block>>>(d_f, d_solid, nx, ny, omega, (step % 2) == 0);
}
```

**三個實作上的重點**：

1. **奇數步會寫到鄰居的記憶體。** 但每個 thread 寫的位址集合互不重疊（因為它讀了鄰居的 `q` 並寫回鄰居的 `q̄`，而鄰居那個 thread 處理的是不同的 `q` 組合）。**這件事值得你自己在紙上把 D1Q3 的情況畫一遍確認**——如果會重疊就需要原子操作，那整個方案就垮了。
2. **不能加 `__restrict__`。** 同一個指標同時讀寫，加了是未定義行為，編譯器可能重排出錯誤的結果。

## L5 — D3Q19 手寫 CUDA + MRT 碰撞

**時間估計：3 週（45–60 小時）**

最陡的一段。前面四個里程碑的東西在這裡全部同時用上，而且多了兩個新戰場：**register 壓力**和 **MRT 的矩陣運算**。

### 學習目標

- 寫出正確的 D3Q19 MRT kernel，與 repo 的 taichi 版逐格一致
- 理解 register 壓力與 occupancy 的取捨，並量出取捨點
- 知道 19 個 moment 分別是什麼、哪些是剪應力、哪些是 ghost
- 理解為什麼 LES 只改剪應力那幾個鬆弛率

### 關鍵觀念

#### D3Q19 速度集

19 個方向 = 1（靜止）+ 6（面心）+ 12（邊心）。**直接照抄 repo 的編號**（`Single_phase/LBM_3D_SinglePhase_Solver.py` 的 `static_init()`）：

| q | c_q | 權重 | q | c_q | 權重 |
| --- | --- | --- | --- | --- | --- |
| 0 | (0,0,0) | 1/3 | 10 | (-1,1,0) | 1/36 |
| 1 | (1,0,0) | 1/18 | 11 | (1,0,1) | 1/36 |
| 2 | (-1,0,0) | 1/18 | 12 | (-1,0,-1) | 1/36 |
| 3 | (0,1,0) | 1/18 | 13 | (1,0,-1) | 1/36 |
| 4 | (0,-1,0) | 1/18 | 14 | (-1,0,1) | 1/36 |
| 5 | (0,0,1) | 1/18 | 15 | (0,1,1) | 1/36 |
| 6 | (0,0,-1) | 1/18 | 16 | (0,-1,-1) | 1/36 |
| 7 | (1,1,0) | 1/36 | 17 | (0,1,-1) | 1/36 |
| 8 | (-1,-1,0) | 1/36 | 18 | (0,-1,1) | 1/36 |
| 9 | (1,-1,0) | 1/36 |  |  |  |

反向表（repo 的 `LR`）：`{0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17}`。 可以看出編號是成對的：奇偶相鄰互為反向（1↔2、3↔4、…），只有 0 是自己。**這個規律讓 `OPP[q]` 可以用算的**：`q == 0 ? 0 : (q % 2 ? q + 1 : q - 1)`，不過查表更清楚，而且 `__constant__` 查表幾乎免費。

#### MRT：它在做什麼

BGK 用一個鬆弛時間 τ 把所有 19 個分量拉向平衡。MRT（Multiple-Relaxation-Time）先把 19 個分布函數轉換成 19 個**矩（moment）**——每個矩有物理意義——再用**各自不同的速率**鬆弛，最後轉回來：

```latex
\mathbf{m} = \mathbf{M}\,\mathbf{f}
```

```latex
\mathbf{m}^{*} = \mathbf{m} - \mathbf{S}\,(\mathbf{m} - \mathbf{m}^{\text{eq}})
```

```latex
\mathbf{f}^{*} = \mathbf{M}^{-1}\,\mathbf{m}^{*}
```

`S` 是對角矩陣（repo 裡的 `S_dig`），所以中間那步是逐元素乘，不是矩陣乘。

**為什麼要 MRT**：BGK 在高 Reynolds 數（τ 接近 0.5）時會數值不穩定，而且固壁附近的黏滯係數會跟網格解析度相依（非物理）。MRT 可以把「影響黏滯係數的矩」和「純數值雜訊的矩（ghost modes）」分開，給後者一個很強的阻尼，穩定性大幅提升。**對建築風環境（Re 動輄 10^5 以上）這是必要的，不是選配。**

#### 19 個 moment 的排序與意義

以下是 repo 那份 `M_np` 對應的排序（d'Humières 系族的一種）：

| index | 矩 | 物理意義 | 類別 | 鬆弛率（repo） |
| --- | --- | --- | --- | --- |
| 0 | ρ | 密度 | **守恆** | 0 |
| 1 | e | 能量 | 體積黏滯 | s_v |
| 2 | ε | 能量平方 | ghost | s_v |
| 3 | j_x | x 動量 | **守恆** | 0 |
| 4 | q_x | x 熱通量 | ghost | s_other |
| 5 | j_y | y 動量 | **守恆** | 0 |
| 6 | q_y | y 熱通量 | ghost | s_other |
| 7 | j_z | z 動量 | **守恆** | 0 |
| 8 | q_z | z 熱通量 | ghost | s_other |
| **9** | 3p_xx | **法向偏應力** | **剪應力** | **s_v** |
| 10 | 3π_xx | 9 的高階對應 | ghost | s_v |
| **11** | p_ww | **法向偏應力** | **剪應力** | **s_v** |
| 12 | π_ww | 11 的高階對應 | ghost | s_v |
| **13** | p_xy | **剪應力 xy** | **剪應力** | **s_v** |
| **14** | p_yz | **剪應力 yz** | **剪應力** | **s_v** |
| **15** | p_xz | **剪應力 xz** | **剪應力** | **s_v** |
| 16 | m_x | 三階 | ghost | s_other |
| 17 | m_y | 三階 | ghost | s_other |
| 18 | m_z | 三階 | ghost | s_other |

平衡矩（repo 的 `meq_vec`，非零項只有這些）：

```latex
m^{eq}_0 = \rho,\quad m^{eq}_1 = \mathbf{u}\cdot\mathbf{u},\quad m^{eq}_3 = u_x,\quad m^{eq}_5 = u_y,\quad m^{eq}_7 = u_z
```

```latex
m^{eq}_9 = 2u_x^2 - u_y^2 - u_z^2,\quad m^{eq}_{11} = u_y^2 - u_z^2
```

```latex
m^{eq}_{13} = u_x u_y,\quad m^{eq}_{14} = u_y u_z,\quad m^{eq}_{15} = u_x u_z
```

其餘 10 個平衡矩為 0。**這表示 ghost modes 的平衡值是零——它們純粹是要被抑制的數值雜訊。**

鬆弛率（repo）：

```latex
s_v = \frac{1}{\tau},\qquad s_{\text{other}} = \frac{8(2 - s_v)}{8 - s_v}
```

`s_other` 這個公式的來源是「magic parameter」條件 Λ = 3/16，它讓 bounce-back 邊界的位置恰好落在固體與流體格點的中點，與 τ 無關。**這是 MRT 相對 BGK 最實際的好處之一**，對建築幾何的準確度很重要。

#### 三個需要誠實標註的觀察

我讀了 repo 的程式碼，有三點你在移植時應該自己驗證，**不要假設 repo 一定是對的，也不要假設我一定是對的**：

## L6 — FP16 儲存 + Esoteric Pull

**時間估計：2 週（30–40 小時）**

**這個里程碑是整條學習軌的決策關鍵。** 如果 FP16 + 單陣列真的帶來 2 倍以上的吞吐和 4 倍的域大小，那「整包換成 CUDA」這個結局就有了充分理由——因為 taichi 對這兩件事的支援目前遠不如手寫。如果沒有，那 taichi 留著。

### 學習目標

- 理解 FP16 的表示範圍與精度，並知道為什麼 LBM 的 DDF 剛好適合
- 實作「儲存 FP16、計算 FP32」的模式，驗證精度可接受
- 概念上理解 Esoteric Pull 為什麼同時達成單陣列、隱式 bounce-back、完美 coalescing

### 關鍵觀念

#### FP16 的表示範圍與精度

IEEE 754 half precision（`__half`，`cuda_fp16.h`）：

```
[1 符號][5 指數][10 尾數]  共 16 bits
```

| 性質 | 數值 |
| --- | --- |
| 最大正規值 | 65504 |
| 最小正規值 | 6.10 × 10⁻⁵ |
| 最小次正規值 | 5.96 × 10⁻⁸ |
| **相對精度（機器 epsilon）** | 2⁻¹¹ ≈ **4.88 × 10⁻⁴ ≈ 0.05%** |
| 有效十進位位數 | 約 3.3 位 |

（對照：FP32 是 1-8-23，相對精度 2⁻²⁴ ≈ 6×10⁻⁸。）

**0.05% 的相對誤差對 LBM 可以接受嗎？** 關鍵在於 DDF 的數值範圍很窄。在無因次化的 LBM 裡，ρ ≈ 1，u ≲ 0.1，所以：

```latex
f_q \approx w_q \rho \approx w_q \in \left[\tfrac{1}{36}, \tfrac{1}{3}\right] \approx [0.028,\ 0.333]
```

全部落在 FP16 的舒適區——離下溢（6×10⁻⁵）和上溢（65504）都很遠。這是 LBM 適合 FP16 的結構性理由。

**但有一個陷阱**：非平衡部分 `f_q - f_q^eq` 才是攜帶物理資訊的部分，而它的量級是 `O(Ma²) ≈ 10⁻²` 甚至更小，相對於 `f_q ≈ 0.1`。用 FP16 存 `f_q`，等於用 0.05% 的精度去解析一個只佔 1% 的擾動——**有效精度只剩約 5%**。

**兩個對策**：

- **只存非平衡部分**：把 `f_q - w_q`（減掉靜止平衡值）用 FP16 存，動態範圍就被重新置中了。這是實務上常見且有效的技巧。
- **自訂 FP16 格式**：Lehmann 等人（*Phys. Rev. E* 106, 015308, 2022，"Accuracy and performance of the lattice Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats"）提出把指數位數縮短、尾數位數加長的自訂 16-bit 格式（論文裡稱 FP16C），因為 DDF 根本用不到 5 bits 的指數範圍。**具體的位元分配請查該篇論文，我不憑記憶寫。** 論文的結論是自訂格式在同樣 16 bits 下精度明顯優於 IEEE half，而且對多數案例的誤差可忽略。

#### 儲存 FP16、計算 FP32

**核心原則：只有進 DRAM 的那一刻才是 FP16，所有算術都在 FP32 做。**

```cuda
#include <cuda_fp16.h>

__global__ void lbm_fp16(const __half* __restrict__ src,
                         __half* __restrict__ dst, ...)
{
    ...
    float f[19];
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        __half h = src[q * N + nb];
        f[q] = __half2float(h);          // 進 register 立刻轉 fp32
    }

    // ... 所有碰撞運算用 float，跟 L5 一字不改 ...

    #pragma unroll
    for (int q = 0; q < 19; ++q)
        dst[q * N + cell] = __float2half(f[q]);   // 出去才轉回 fp16
}
```

**為什麼不用 FP16 算？**

- 精度會在 19 項加總裡累積誤差，得不償失
- 算術根本不是瓶頸（見開頭那節），FP16 算術省的時間是零
- FP32 算術在所有卡上都有全速支援，FP16 標量算術反而在某些卡上更慢

**`__half2`（一次處理兩個 half）值得嗎？**

`__half2` 是把兩個 half 打包成 32 bits，有 SIMD 風格的指令（`__hadd2`、`__hmul2`、`__half22float2`）。對 LBM：

- **載入/儲存**值得：`__half2` 的載入是 4 bytes，跟 `float` 一樣，coalescing 行為比逐個 `__half`（2 bytes）好。一個 warp 讀 32 個 `__half` 只有 64 bytes = 2 sectors，雖然仍然合併，但每次交易搬的有效資料少。用 `__half2` 一次拿兩個 q 或兩個相鄰 cell，效率更好。
- **算術**不值得，理由同上。

**實作建議**：先用最樸素的逐個 `__half` 版本跑通、驗證精度，再試 `__half2` 的向量化載入看有沒有差。**不要一開始就兩件事一起做**，否則出問題分不清是精度還是索引。

**另一個選項：`__nv_bfloat16`（`cuda_bf16.h`）。** 1-8-7 格式，指數範圍跟 FP32 一樣寬，但尾數只有 7 bits → 相對精度 2⁻⁸ ≈ 0.4%，**比 FP16 差 8 倍**。bfloat16 是為深度學習設計的（要寬範圍不要精度），**對 LBM 是錯的選擇**。知道它存在，然後不要用。

#### 精度驗證的方法論

FP16 版本**不可能**跟 FP32 版逐格一致，所以 L2 那套驗證要換掉。改用：

1. **短時程統計比對**：跑 100 步，比 FP32 版與 FP16 版的速度場。預期 `max rel diff ~ 1e-3`，`L2 norm of diff / L2 norm of field < 1e-3`。
2. **物理量收斂**：lid-driven cavity 跑到穩態，中心線剖面對 Ghia 表格。**FP16 版的誤差應該跟 FP32 版是同一個數量級**——如果 FP32 版誤差 0.5%、FP16 版 0.6%，那 FP16 是可用的；如果 FP16 版變成 5%，就不可用。
3. **長時程穩定性**：跑 10⁵ 步看會不會漂移或爆掉。這是 FP16 最大的風險——小誤差可能在長時間積分下累積。**這一項比單步精度重要得多。**
4. **質量守恆**：FP16 下總質量會緩慢漂移。量出漂移率（每步相對變化），判斷在你的目標模擬長度下是否可忽略。

## L7 — 與 taichi 版完整 benchmark 對照與決策

**時間估計：1 週（15–20 小時）**

這一週的產出不是程式碼，是**一份能讓你（和你的口試委員、合作者）信服的決策依據**。

### 學習目標

- 建立一套可重現的 benchmark 方法論
- 用 Nsight Compute 判讀「是否已接近頻寬上限」
- 對三種結局做出有數據支撐的選擇

### Benchmark 方法論

#### 必須做的六件事

**（1）暖機（warm-up）**

第一次呼叫 kernel 時會發生：CUDA context 建立（可能數百 ms）、JIT 編譯（taichi 一定有，CuPy 的 NVRTC 也有）、GPU 從低功耗狀態升頻、cache 冷。

```python
for _ in range(100):      # 暖機，不計時
    step()
sync()                    # 一定要同步
t0 = time.perf_counter()
for _ in range(1000):     # 計時
    step()
sync()
t1 = time.perf_counter()
```

**taichi 的 JIT 特別要注意**：taichi 在第一次執行 `@ti.kernel` 時才編譯，而且 offline cache 可能讓第二次執行快很多。**確保暖機跑了足夠多步（至少 100），並且 taichi 和 CUDA 兩邊用同樣的暖機步數。**

**（2）同步**

`sync()` 在 CUDA 是 `cudaDeviceSynchronize()`，在 taichi 是 `ti.sync()`。**沒有同步的計時結果毫無意義**——你量到的是 launch queue 的速度。

**（3）重複與統計**

跑 5 次完整量測，**取中位數，並回報最小值與最大值**。不要取平均（會被偶發的系統干擾污染）。若 max/min > 1.1，代表環境不穩定（背景程序、降頻），先處理環境。

**（4）時脈與熱穩定**

```bash
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv -l 1
```

跑 benchmark 的同時監看。如果 SM 時脈在過程中掉了，你量的是降頻後的效能。有權限的話：`sudo nvidia-smi -pm 1` 開持續模式、`sudo nvidia-smi -lgc <min>,<max>` 鎖時脈。**至少要在每次量測前讓 GPU 冷卻回到同一個起點。**

**（5）排除 I/O 與後處理**

計時迴圈裡不能有 VTK 輸出、`to_numpy()`、`print`。這些會強制同步並加上巨大的常數。

**（6）公平性檢查清單**

這是最容易搞砸的一環。兩邊必須對齊：

- [ ] 同樣的域大小（含 padding 策略）
- [ ] 同樣的碰撞模型（都是 MRT，用同一個 `M`）
- [ ] 同樣的精度（都是 f32，或都是 f64）
- [ ] 同樣的邊界條件
- [ ] **同樣的 kernel 融合程度**（如果 CUDA 版是融合的單 kernel，taichi 版也要改成融合的單 kernel）
- [ ] **同樣的資料佈局**（如果 CUDA 用 SoA，taichi 要用 `ti.root.dense(...).place(...)` 明確指定 SoA）
- [ ] 同樣的暖機與計時步數
- [ ] 同一張卡、同一個 session

**第五、六項是關鍵。** 如果你拿「融合 + SoA 的 CUDA」比「四個分開 kernel + 預設佈局的 taichi」，你量到的是**你自己做的演算法改良**，不是 CUDA vs taichi。這會導出錯誤的決策（換掉 taichi），而正確的決策可能是「改良 taichi 版就好」。

**建議的比較矩陣**：

| 版本 | MLUPS | % 理論上限 | 記憶體 (GB) |
| --- | --- | --- | --- |
| taichi，原始（4 kernel、預設佈局） |  |  |  |
| taichi，融合 + SoA |  |  |  |
| CUDA，L5（融合 + SoA + f32 雙陣列） |  |  |  |
| CUDA，L6（FP16 + in-place） |  |  |  |

**第 1 列到第 2 列的差距 = 演算法改良的價值（可以不換語言就拿到）。 第 2 列到第 3 列的差距 = 換語言的真實價值。 第 3 列到第 4 列的差距 = FP16 + in-place 的價值（目前只有手寫拿得到）。**

這張四列的表就是你的決策依據。**把它做出來，這週就成功了。**

### Profiling

#### Nsight Compute 的關鍵指標

```bash
ncu --set full -o profile_report ./lbm_cuda        # 完整報告，慢
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy ./lbm_cuda
```

要看的四個東西，**按重要性排序**：

| 指標 | 在哪 | 意義 | LBM 的目標值 |
| --- | --- | --- | --- |
| **DRAM throughput %** | SpeedOfLight 區段的 "Memory [%]" | 佔記憶體峰值頻寬的百分比 | **> 80% = 成功** |
| **Sectors per request** | MemoryWorkloadAnalysis | coalescing 品質 | f32: 4；f16: 2 |
| **Achieved occupancy** | Occupancy 區段 | 實際駐留 warp 比例 | 25–50% 就夠（見 L5） |
| **Register per thread / spills** | Launch Statistics | register 壓力 | spill = 0 |

**判讀規則**：

- **DRAM > 80%，Compute < 40%** → **你已經打到頻寬牆了。** 剩下唯一的路是減少搬的 bytes（FP16、in-place），不是最佳化程式碼。**這是 L7 最重要的一個判斷。**
- **DRAM < 50%，Compute < 50%** → 兩邊都沒滿，代表是延遲問題。查 spill、查 occupancy、查 divergence。
- **DRAM < 50%，Sectors per request > 8** → coalescing 壞了。回 L3。
- **Compute > 70%** → 對 LBM 來說這很異常。檢查是不是有除法、`expf`、或某個迴圈沒展開。

**版本注意**：Nsight Compute 的 section 名稱與 metric 名稱在各版本間會變動。用 `ncu --list-sections` 和 `ncu --query-metrics` 查你裝的版本，**不要照抄我列的名字**。另外 `--set full` 會把 kernel 重放很多次，profile 時用小域（例如 128³），量效能時才用大域。

#### taichi 這邊的 profiling

## 附節 A — 把 taichi 當成 CUDA 的教學工具

這一節不是里程碑，是**貫穿全程的工作方法**。它可能是整份文件裡最能節省時間的一節。

### 核心想法

taichi 已經是一個 GPU 編譯器。你要手寫的東西，它已經在做了。**與其對著空白檔案想「這該怎麼寫」，不如叫 taichi 先寫一次給你看。**

工作流程：

```
1. 在 taichi 裡寫出「正確」的版本        ← 快，有 Python 的除錯便利
2. 開 print_ir，看它產生了什麼 IR        ← 學到記憶體佈局、迴圈結構、索引算法
3. 開 kernel_profiler，看哪個 kernel 熱  ← 決定該先手寫哪一個
4. 手寫 CUDA 版                          ← 有參考，不是憑空
5. 兩邊逐格比對                          ← taichi 版就是你的 ground truth
```

**第 5 步是關鍵**：你永遠有一個已知正確的參考實作。沒有它，你手寫 CUDA 時的每個 bug 都要從物理直覺去猜。

### `print_ir=True`

```python
ti.init(arch=ti.cuda, print_ir=True)
```

它會把 taichi 的中間表示（IR，Intermediate Representation）印到 stdout。輸出很長，但你要看的只有幾樣東西：

**（1）位址計算 → 記憶體佈局**

找 `GlobalPtrStmt` 或類似的節點，看索引是怎麼組出來的。如果看到類似 `(i*ny + j)*nz + k) * 19 + q` 的形式，那是 AoS；如果是 `q * (nx*ny*nz) + ...`，那是 SoA。**這直接回答了 L3 那個問題。**

**（2）迴圈結構 → 平行化策略**

看 `OffloadedStmt` 的 `range_for` 或 `struct_for`，以及 `block_dim`。這告訴你 taichi 選了多大的 block、怎麼切 grid。

**（3）常數摺疊與迴圈展開**

`ti.static(range(19))` 有沒有真的展開？ `M[None]` 的元素有沒有變成常數？如果沒有，那 taichi 每步都在從記憶體讀那 361 個係數——**那就是一個你的 CUDA 版可以贏它的地方**。

**實用建議**：`print_ir` 的輸出量很大，重導到檔案再看：

```python
ti.init(arch=ti.cuda, print_ir=True)
# python run.py > ir.txt 2>&1
```

而且只對**一個小 kernel、一個小域**開，不然你會被淹沒。

**更底層的輸出**：taichi 還有印出 LLVM IR / PTX 的選項（我印象中是 `print_kernel_llvm_ir=True` 之類的參數名，**但各版本不同，請查 taichi 文件的「Developer utilities」或 `ti.init` 的參數清單，不要照抄我寫的名字**）。看 PTX 能直接知道 register 用量和實際指令，比 taichi IR 更接近真相。

另外，taichi 的 offline cache 目錄裡會留下編譯產物，有時候可以直接找到生成的 CUDA/PTX。**這條路值得花半小時探索一次。**

### `kernel_profiler=True`

```python
ti.init(arch=ti.cuda, kernel_profiler=True)
# ... 暖機 ...
ti.profiler.clear_kernel_profiler_info()
# ... 正式量測 ...
ti.sync()
ti.profiler.print_kernel_profiler_info()
```

用途（按價值排序）：

1. **決定手寫的優先順序。** 佔 45% 的 kernel 值得手寫，佔 2% 的不值得。
2. **建立 taichi 版的 baseline。** L7 的比較表第一列。
3. **驗證你的改良有效。** 把 taichi 版改成融合 + SoA 之後，同一張表能看出改善多少。
4. **抓出意料之外的熱點。** 常見的驚喜：邊界條件 kernel 意外地貴（因為 divergence）、或 `to_numpy()` 每步都在同步。

repo 的 `Single_phase/example_cavity.py` 第 4 行已經有這兩個旗標（目前都設成 `False`），把它們打開就能開始。

### 一個具體的練習流程

在進 L2 之前，花半天做這件事：

1. 拿你的 taichi D2Q9，`ti.init(arch=ti.cuda, print_ir=True)`，域設 32×32，跑一步，把 IR 存起來
2. 在 IR 裡找出：f 的位址演算法、block_dim、迴圈有沒有展開
3. 寫下三句話：「taichi 用了 ___ 佈局」「taichi 選了 block size ___」「19/9 的迴圈 ___ 展開」
4. 把 `print_ir` 關掉、`kernel_profiler` 打開，域設 512×512，跑 1000 步，記下每個 kernel 的時間
5. 算出 taichi 版的 MLUPS 和佔理論上限的百分比

**做完這五步，你對「該不該換 taichi」已經有了初步答案，而你還沒寫一行 CUDA。** 如果第 5 步的答案是「taichi 已經達到 85%」，那後面六個里程碑的性質就從「找出替代方案」變成「確認沒有更好的方案」——這仍然值得做，但心態和時間預算應該不同。

---

## 附節 B — 參考資料與查證清單

### 我在這份文件裡標註為不確定的地方

寫在這裡集中列一次，方便你逐項查證。**這些不是我在偷懶，是我判斷憑記憶寫下來的風險大於價值的地方。**

| 主題 | 該查哪裡 |
| --- | --- |
| taichi `ti.Vector.field` 的預設記憶體佈局 | taichi 文件「Fields (advanced)」/「Layout」；或直接 `print_ir` 實測 |
| taichi profiler 的 API 名稱（版本差異） | 你裝的 taichi 版本的文件「Profiler」一節 |
| taichi 印出 LLVM IR / PTX 的參數名 | `ti.init` 的參數清單、「Developer utilities」 |
| taichi `ti.f16` 在 CUDA backend 的成熟度 | 實測。這直接影響 L7 結局 C 的理由 |
| Nsight Compute 的 section / metric 名稱 | `ncu --list-sections`、`ncu --query-metrics` |
| Lehmann 自訂 FP16C 的確切位元分配 | Lehmann et al., *Phys. Rev. E* 106, 015308 (2022) |
| Esoteric Pull 的索引推導與實作 | Lehmann, *Computation* 10(6), 92 (2022)；FluidX3D 原始碼 |
| Smagorinsky LES 在 MRT 下的閉式解 | Hou et al. (1996)；Krüger et al. 教科書第 17 章 |
| repo 的 `tau_f = niu/3.0 + 0.5` 是否正確 | 用 Poiseuille 解析解反推，實測 |
| repo 的 index 10/12 用 s_v 是否刻意 | 對照 d'Humières et al. (2002)，或實測高 Re 穩定性 |
| CuPy 傳遞 `__half` 陣列的正確做法 | CuPy 文件「User-Defined Kernels」 |
| 各 GPU 的 register/SM、L2 大小、max threads/SM | CUDA C++ Programming Guide 附錄「Compute Capabilities」的表格 |

### 核心參考文獻

**LBM 本身**

- Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva, Viggen, *The Lattice Boltzmann Method: Principles and Practice*, Springer 2017 — **如果只讀一本，讀這本。** 第 13 章講 GPU 實作。
- d'Humières, Ginzburg, Krafczyk, Lallemand, Luo, "Multiple-relaxation-time lattice Boltzmann models in three dimensions", *Phil. Trans. R. Soc. Lond. A* 360, 437 (2002) — MRT 的原始文獻
- Ghia, Ghia, Shin, "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method", *J. Comput. Phys.* 48, 387 (1982) — cavity 驗證的標準表格

**GPU 上的 LBM**

- Lehmann, "Esoteric Pull and Esoteric Push", *Computation* 10(6), 92 (2022)
- Lehmann, Krause, Amati, Sega, Harting, "Accuracy and performance of the lattice Boltzmann method with 64-bit, 32-bit, and customized 16-bit number formats", *Phys. Rev. E* 106, 015308 (2022)
- Bailey, Myre, Walsh, Lilja, Saar, "Accelerating Lattice Boltzmann Fluid Flow Simulations Using Graphics Processors", *ICPP* 2009 — AA-pattern 的原始文獻
- **FluidX3D 原始碼**（GitHub `ProjectPhysX/FluidX3D`）— **最有價值的單一參考。** 它是目前公認最快的開源 LBM，而且程式碼寫得清楚。OpenCL 不是 CUDA，但概念與索引邏輯可以直接搬。

**CUDA**

- *CUDA C++ Programming Guide* — 尤其是 "Performance Guidelines"（coalescing）和附錄 "Compute Capabilities"（各架構的硬體數字表）
- *CUDA C++ Best Practices Guide* — 較短，實用
- *Nsight Compute Documentation* — Kernel Profiling Guide 那一章解釋每個 metric 的意義

**工具**

- CuPy: User-Defined Kernels
- taichi: Fields (advanced)、Profiler、Developer utilities

### 給主線的一句話

這條學習軌走完之後，無論結局是哪一個，你手上會多兩樣主線用得到的東西：

1. **一個獨立的 ground truth 實作**，之後 taichi 版改任何東西都可以拿它驗證
2. **一套可重現的 benchmark 與 profiling 流程**，讓「快了多少」這個問題永遠有答案

這兩樣東西的價值，跟「最後有沒有換掉 taichi」無關。

```
ptxas info: Used 96 registers, 0 bytes cumulative stack size, 376 bytes cmem[0]
```

`0 bytes ... stack size` 表示沒有 spill。看到非零就要警覺。這件事在 L5 會變成主要戰場。

**另一個陷阱：array 索引不是編譯期常數時會強制 spill。**

```cuda
float f[19];
for (int q = 0; q < 19; q++) f[q] = ...;   // 迴圈展開後 q 是常數 → 放 register
// 但：
int q = some_runtime_value;
f[q] = ...;                                 // q 不是常數 → 整個 f[] 掉到 local memory
```

這對應到 taichi 裡的 `dynamic_index` 設定。repo 的 `Single_phase/example_cavity.py` 第 4 行就寫著 `dynamic_index=False`——那正是在叫 taichi 不要產生動態索引、好讓陣列留在 register。你現在知道為什麼了。

#### 三個修飾詞

```cuda
__global__ void kernel(float* a);   // 從 host 呼叫，在 device 執行。回傳值必須是 void。
__device__ float helper(float x);   // 從 device 呼叫，在 device 執行。≈ taichi 的 ti.func
__host__   void normal(void);       // 一般 CPU 函式（預設，通常省略）
__host__ __device__ float both(float x);  // 兩邊都編一份，共用程式碼時很好用
```

對應關係，給有 taichi 背景的人：

| taichi | CUDA | 備註 |
| --- | --- | --- |
| `@ti.kernel` | `__global__` | taichi kernel 自動有最外層迴圈平行化；CUDA 要自己算索引 |
| `@ti.func` | `__device__`（通常自動 inline） | taichi 的 ti.func 一定 inline，CUDA 可加 `__forceinline__` |
| `ti.field` | `cudaMalloc` 出來的指標 | taichi 管生命週期，CUDA 你自己管 |
| `ti.static(range(19))` | `#pragma unroll` 或寫死的迴圈 | 兩者都是為了讓索引變編譯期常數 |
| `ti.sync()` | `cudaDeviceSynchronize()` |  |
| （沒有對應） | `__syncthreads()` | block 內同步，taichi 層級沒有暴露 |

#### Kernel launch 語法與同步

```cuda
kernel<<<grid, block, shared_bytes, stream>>>(args...);
```

後兩個參數可省略（預設 0 和 default stream）。**關鍵：kernel launch 是非同步的。** 這行執行完，kernel 可能還沒開始跑。所以：

- 計時**一定**要在前後加同步，或用 CUDA event
- kernel 裡的錯誤不會在 launch 那行回報，要在同步點才看得到

#### 錯誤檢查的正確寫法

這是新手最常省略、然後浪費最多時間的東西。先寫好這個巨集，之後每個 CUDA API 呼叫都包起來：

```cuda
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d — %s (%s)\n",             \
                    __FILE__, __LINE__,                                    \
                    cudaGetErrorString(err_), #call);                      \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// kernel launch 沒有回傳值，要另外檢查兩件事：
#define CUDA_CHECK_KERNEL()                                                \
    do {                                                                   \
        CUDA_CHECK(cudaGetLastError());      /* launch 設定錯誤，例如 block 太大 */ \
        CUDA_CHECK(cudaDeviceSynchronize()); /* 執行期錯誤，例如越界存取 */     \
    } while (0)
```

`do { ... } while(0)` 的包裝是為了讓巨集在 `if (x) CUDA_CHECK(...); else ...` 這種情境下語法正確。

**注意**：`CUDA_CHECK_KERNEL()` 裡的 `cudaDeviceSynchronize()` 會強制等待，在正式跑的時候會拖慢效能。實務做法是用編譯旗標控制，debug build 才開。

越界寫入這種錯誤，`cudaDeviceSynchronize()` 通常只會給你一個很不具體的 `an illegal memory access was encountered`。這時候用：

```bash
compute-sanitizer ./my_program
```

它會直接指出哪一行、哪個 thread 越界。（舊版 CUDA 叫 `cuda-memcheck`，CUDA 11 之後改名 `compute-sanitizer`。）**這個工具值得你在 L2 之前就先學會用**，它能省掉幾天的痛苦。

#### 編譯

```bash
nvcc -O3 -arch=sm_86 -lineinfo -Xptxas -v vector_add.cu -o vector_add
```

| 旗標 | 作用 |
| --- | --- |
| `-arch=sm_86` | 目標 compute capability。RTX 30 系是 86、RTX 40 系是 89、A100 是 80、H100 是 90。用 `nvidia-smi --query-gpu=compute_cap --format=csv` 查你的卡。 |
| `-lineinfo` | 產生行號對應，Nsight Compute 才能把效能數字對應回原始碼行。**L7 會需要，建議一開始就加。** |
| `-Xptxas -v` | 印出 register 用量和 spill 資訊 |
| `-O3` | 最佳化 |
| `--use_fast_math` | **先不要加。** 它會把除法、平方根換成低精度快速版本，讓你的 L2 數值比對失敗，你會分不清是 bug 還是精度。等驗證通過再考慮。 |

#### 從 Python 呼叫

最終你要把 kernel 接回 Grasshopper 的 Python 流程，所以這件事要早點試。三種做法：

**（A）CuPy `RawKernel` — 建議從這個開始**

```python
import cupy as cp

src = r'''
extern "C" __global__
void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
'''
# 注意 extern "C"：不加的話 C++ name mangling 會讓名字對不上
ker = cp.RawKernel(src, 'vector_add')

n = 1 << 20
a = cp.random.rand(n, dtype=cp.float32)
b = cp.random.rand(n, dtype=cp.float32)
c = cp.empty_like(a)

block = 256
grid = (n + block - 1) // block
ker((grid,), (block,), (a, b, c, n))      # 注意：grid/block 是 tuple

cp.testing.assert_allclose(c, a + b)
```

- 優點：沒有獨立的 build step，NVRTC 在執行期編譯並快取；CuPy 陣列跟 numpy 幾乎同介面；後處理、VTK 輸出都能直接接
- 缺點：需要裝對應 CUDA 版本的 CuPy wheel；編譯錯誤訊息有時候不太好讀
- 多個 kernel 共用 `__device__` 函式時改用 `cp.RawModule(code=src, options=('-std=c++17',))`，再用 `.get_function(name)` 取出

**（B）ctypes + 自己編 .so**

```bash
nvcc -O3 -arch=sm_86 -Xcompiler -fPIC -shared lbm.cu -o liblbm.so
```

```python
import ctypes
lib = ctypes.CDLL('./liblbm.so')
lib.lbm_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.lbm_step.restype = None
# 傳指標：CuPy 陣列用 arr.data.ptr，PyTorch tensor 用 t.data_ptr()
lib.lbm_step(a.data.ptr, b.data.ptr, n)
```

- 優點：完全控制編譯旗標；最少的執行期相依；部署到 Rhino 的 CPython 環境時最不容易出問題
- 缺點：要自己寫 C 介面層、自己管型別；跨平台要各編一份
- **對 Grasshopper 整合這條路特別值得評估**：Rhino 8 內建 CPython，裝額外 wheel 有時候會卡在權限或版本上，而載入一個 .so 幾乎不會

**（C）PyCUDA `SourceModule`**

歷史比較久，教材多。但 context 管理要自己來、跟現代 Python GPU 生態（CuPy / PyTorch / taichi 的 CUDA context）共存時容易踩雷。**如果是新開始，建議跳過，直接用 A 或 B。**

（另外還有 `torch.utils.cpp_extension.load_inline` 和 Numba `@cuda.jit`。前者好用但拖一整包 PyTorch；後者是 Python 語法寫 CUDA，對這條學習軌來說反而繞路——你要學的正是 C++ 那一層。）

### 具體練習

1. **環境確認**：`nvidia-smi`、`nvcc --version`、查出你的卡的 compute capability、跑 CUDA samples 裡的 `deviceQuery` 和 `bandwidthTest`。**把 `bandwidthTest` 的 Device-to-Device 數字抄下來，這是你之後所有理論上限計算的分母。** 它通常是規格值的 80–90%。
2. **Vector add**：純 C++ 檔案版本，含 `CUDA_CHECK`，用 `cudaEvent` 計時。
3. **算出你的 vector add 達到頻寬的幾 %**：

```latex
\text{有效頻寬} = \frac{3 \times N \times 4\ \text{bytes}}{t}
```

（讀 a、讀 b、寫 c，共三次）。**這題應該能達到 `bandwidthTest` 數字的 85% 以上。達不到就是哪裡寫錯了，先解決再往下走。**

1. **故意寫壞**：把索引改成 `i * 32`（製造 uncoalesced 存取），量一次；把 block size 改成 33（非 32 倍數），量一次；把 block size 改成 1024 和 32 各量一次。記下四組數字——這是你對 CUDA 效能的第一手直覺。
2. **Python 綁定**：用上面 A 或 B 的其中一種，把同一個 kernel 從 Python 叫起來，確認結果與 numpy 一致。
3. **跑一次 `compute-sanitizer`**：故意把邊界守衛拿掉，看它怎麼報錯。

### 驗證方式

- vector add 結果與 `numpy` 逐元素比對，f32 應該是**完全相等**（`np.array_equal`），因為加法沒有重排空間
- 有效頻寬 ≥ `bandwidthTest` 的 85%
- `compute-sanitizer` 零錯誤
- `-Xptxas -v` 顯示 `0 bytes stack size`

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| kernel 好像沒跑，結果全 0 | 忘了 `cudaMemcpy` 回 host，或忘了 `cudaDeviceSynchronize` 就讀結果 |
| 結果只有前面一段正確 | grid 大小算錯，用了 `n / block` 而不是 `(n + block - 1) / block` |
| `invalid device function` | `-arch` 跟實體卡不符 |
| CuPy 找不到 kernel 名字 | 忘了 `extern "C"` |
| 時間量出來是 0 或荒謬地小 | 沒同步就計時，量到的是 launch 的時間不是執行的時間 |
| 改了 kernel 但行為沒變 | CuPy 的 NVRTC 快取。改 source 通常會失效重編，但若你只改了 include 的檔案就不會——重啟 Python |
| 偶發的錯誤結果 | 幾乎一定是競爭條件或越界。`compute-sanitizer` |

---

```cuda
// Pull 的骨架
__global__ void lbm_step(const float* __restrict__ src,
                         float* __restrict__ dst,
                         const uint8_t* __restrict__ solid,
                         int nx, int ny, float omega)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const int cell = y * nx + x;

    if (solid[cell]) return;               // 固體節點，跳過

    // ---- 1. Pull：從各個方向的上游鄰居讀 ----
    float f[9];
    #pragma unroll
    for (int q = 0; q < 9; ++q) {
        // 上游 = 沿 -c_q 方向退一格（週期邊界用取模）
        int xs = (x - CX[q] + nx) % nx;
        int ys = (y - CY[q] + ny) % ny;
        f[q] = src[q * nx * ny + ys * nx + xs];
    }

    // ---- 2. 巨觀量 ----
    float rho = 0.0f, ux = 0.0f, uy = 0.0f;
    #pragma unroll
    for (int q = 0; q < 9; ++q) {
        rho += f[q];
        ux  += CX[q] * f[q];
        uy  += CY[q] * f[q];
    }
    ux /= rho;  uy /= rho;

    // ---- 3. BGK 碰撞 ----
    const float usq = ux * ux + uy * uy;
    #pragma unroll
    for (int q = 0; q < 9; ++q) {
        const float cu = 3.0f * (CX[q] * ux + CY[q] * uy);
        const float feq = W[q] * rho * (1.0f + cu + 0.5f * cu * cu - 1.5f * usq);
        f[q] += omega * (feq - f[q]);      // omega = 1/tau
    }

    // ---- 4. 寫回自己 ----
    #pragma unroll
    for (int q = 0; q < 9; ++q)
        dst[q * nx * ny + cell] = f[q];
}
```

常數表用 `__constant__`，它有專屬 cache，而且 warp 內所有 thread 讀同一個位址時是**廣播**，幾乎免費：

```cuda
__constant__ int   CX[9] = { 0, 1, 0,-1, 0, 1,-1,-1, 1};
__constant__ int   CY[9] = { 0, 0, 1, 0,-1, 1, 1,-1,-1};
__constant__ float W [9] = {4.f/9,  1.f/9, 1.f/9, 1.f/9, 1.f/9,
                            1.f/36, 1.f/36, 1.f/36, 1.f/36};
// 從 host 端設定用 cudaMemcpyToSymbol(CX, host_cx, sizeof(host_cx));
```

`__restrict__` 告訴編譯器 `src` 和 `dst` 不重疊，它才敢做積極的載入重排。**忘了加會損失可觀的效能，成本卻是零。** 注意：到了 L4 的 in-place 版本，src 和 dst 是同一塊記憶體，**那時候就不能加 `__restrict__`**，否則是未定義行為。

#### 邊界處理

三種處理方式，複雜度遞增：

**（1）週期邊界（periodic）**：上面那段用 `%` 取模。簡單但 `%` 在 GPU 上是整數除法，不便宜。快的寫法：

```cuda
int xs = x - CX[q];
xs = (xs < 0) ? nx - 1 : ((xs >= nx) ? 0 : xs);   // 分支比取模快
```

或者邊界只差一格時，用加 nx 再減：`xs = (x - CX[q] + nx) - ((x - CX[q] + nx >= nx) ? nx : 0);`

**（2）Halfway bounce-back（半路反彈，無滑移固壁）**：這是建築風環境最常用的。Pull 版本的寫法很優雅：

```cuda
int xs = x - CX[q], ys = y - CY[q];
if (xs < 0 || xs >= nx || ys < 0 || ys >= ny || solid[ys*nx+xs]) {
    // 上游是固體或域外：改讀自己的反向分量（bounce-back）
    f[q] = src[OPP[q] * nx * ny + cell];
} else {
    f[q] = src[q * nx * ny + ys * nx + xs];
}
```

`OPP[]` 就是 repo 裡的 `LR[]` 那張表。D2Q9 版：`{0, 3, 4, 1, 2, 7, 8, 5, 6}`（依你自己的編號順序調整）。

**（3）速度/壓力入出口**：Zou-He 或 non-equilibrium bounce-back。**L2 不要碰這個。** 先用週期 + bounce-back 把主 kernel 驗證通過，入出口留到後面。

**Halo（暈）技巧**：在四周多加一圈節點，標記成固體，就不用在內圈 kernel 裡做邊界判斷。域從 `nx × ny` 變成 `(nx+2) × (ny+2)`，換掉每個 thread 的分支。對 LBM 是值得的，也讓程式碼乾淨很多。**建議一開始就這樣配置。**

### 驗證方式（這節最重要）

不要用「看起來像流體」當驗證。建立這個四層階梯：

#### 第 0 層：相同初始狀態

讓兩邊從**位元完全相同**的初始 DDF 出發。taichi 那邊：

```python
# taichi 版
solver.init_simulation()
f0 = solver.f.to_numpy()          # shape (nx, ny, 9) 或類似
np.save('f_init.npy', f0)
```

CUDA 那邊從同一個 `.npy` 載入。**這一步做了，後面才有意義**；否則你永遠分不清差異來自初始條件還是 kernel。

#### 第 1 層：單步、逐格、緊容差

跑**一步**，兩邊都 dump 出 f，逐元素比：

```python
fa = taichi_one_step(f0)     # (nx, ny, 9)
fb = cuda_one_step(f0)
d  = np.abs(fa - fb)
r  = d / (np.abs(fa) + 1e-30)
print('max abs diff  :', d.max())
print('max rel diff  :', r.max())
print('argmax location:', np.unravel_index(d.argmax(), d.shape))
```

**容差建議（f32，單步）**：

- `max rel diff < 1e-6` → 正確，差異來自浮點加總順序不同
- `1e-6 ~ 1e-4` → 可疑。檢查是不是有 `--use_fast_math`、或某處用了 `expf`/除法的近似版本
- `> 1e-4` → **有 bug**，不要往下走

**更強的驗證**：兩邊都改成 f64 跑一步。此時 `max rel diff` 應該落在 `1e-14` 量級。如果 f64 下一致、f32 下不一致，那就真的只是精度；如果 f64 下也不一致，就是邏輯 bug。**這個技巧能把「是 bug 還是精度」這個問題一刀切開，非常值得花時間做。**

`argmax location` 會直接告訴你 bug 在哪：如果最大差異都在邊界上，是邊界條件寫錯；如果均勻散佈在內部，是碰撞或 streaming 寫錯；如果集中在某個 `q`，是那個方向的索引算錯。

#### 第 2 層：短時程累積

跑 100 步。f32 的差異會隨機漫步式累積，預期 `max rel diff` 在 `1e-5 ~ 1e-4`。如果是指數成長（每 10 步大一個數量級），那是不穩定或真 bug。

#### 第 3 層：物理驗證

超過幾百步之後**不要再逐格比**——混沌系統裡任何微小差異都會發散，這是物理不是 bug。改比物理量：

- **總質量守恆**：`rho.sum()` 隨時間的相對變化應 < 1e-6（純週期邊界下應該是機器精度）
- **Lid-driven cavity（頂蓋驅動空穴）**：跑到穩態，取中心線速度剖面，跟 Ghia, Ghia & Shin (1982) 的表格比。Re=100 和 Re=400 在合理解析度（128² 以上）下應該貼得很近。**這是 LBM 界的標準驗證案例，一定要做。**
- **Poiseuille flow（管流）**：有解析解（拋物線剖面），可算 L2 誤差，還能驗證你的 bounce-back 邊界是不是真的落在半格位置
- **Taylor-Green vortex**：有解析衰減率，可以驗證你的黏滯係數跟 tau 的關係對不對

repo 裡已經有 `example_cavity.py` 和 `example_poiseuille_flow.py` 可以當 taichi 端的參考。

### 具體練習

1. 建立 `f_init.npy` 交換機制，兩邊都能讀寫
2. 寫融合 Pull kernel，SoA 佈局，週期邊界，BGK
3. 通過第 1 層驗證（單步 f64 比對）
4. 加入 halfway bounce-back 與固體 mask，通過第 1 層驗證
5. 跑 lid-driven cavity Re=100 到穩態，畫中心線剖面對 Ghia
6. 量 MLUPS，跟理論上限比
7. **把 kernel 包成 CuPy RawKernel，從 Python 驅動整個迴圈**——這一步別拖到最後，它決定了主線能不能接得上

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| 一步之後差異就是 1e-2 量級 | 方向編號對不上。印出兩邊的 `c[q]` 和 `w[q]` 表逐項比對 |
| 差異只在邊界一圈 | bounce-back 的方向反了，或 `OPP[]` 表錯了 |
| 跑幾百步後 NaN | tau 太接近 0.5（黏滯係數太小），或速度超過 Mach 限制（u > 0.1 就要小心） |
| 流場對但質量緩慢流失 | 固體節點被計入了總和，或 bounce-back 寫回了域外 |
| Push 版本結果不確定 | 多個 thread 寫同一格。這正是選 Pull 的原因 |
| taichi 的 `to_numpy()` 形狀跟預期不同 | vector field 的 component 維度位置。印出 `.shape` 確認，不要猜 |
| 融合 kernel 比兩個分開的 kernel 還慢 | register 用量爆掉導致 spill。看 `-Xptxas -v` |
| 明明沒改 kernel，MLUPS 每次跑差很多 | 沒暖機（見 L7），或 GPU 在降頻。用 `nvidia-smi -q -d CLOCK` 看 |

---

| GPU | 頻寬 (GB/s) | D3Q19 f32 雙陣列<br>152 B → MLUPS 上限 | D2Q9 f32 雙陣列<br>72 B → MLUPS 上限 |
| --- | --- | --- | --- |
| RTX 3060 12GB | 360 | ~2 370 | ~5 000 |
| RTX 4070 | 504 | ~3 320 | ~7 000 |
| RTX 3090 | 936 | ~6 160 | ~13 000 |
| RTX 4090 | 1 008 | ~6 630 | ~14 000 |
| A100 40GB | 1 555 | ~10 230 | ~21 600 |
| H100 SXM | ~3 350 | ~22 000 | ~46 500 |

（這些是規格值除以 bytes，實際可達約 80–90%。**請用你自己的卡實測值重算一次，並把這張表換成你自己的。**）

**怎麼用這張表**：你的 CUDA 版跑出 5 200 MLUPS、卡是 4090、理論上限 6 630——那你達到 78%，還有一點空間但不多。跑出 1 800 MLUPS——那是 27%，一定有東西壞掉，八成是 coalescing。

### 具體練習：設計一個乾淨的實驗

乾淨的意思是：**只有佈局這一個變因不同，其他全部相同。**

```cuda
// 用編譯期開關切換佈局，其餘程式碼一字不改
#ifdef USE_AOS
  #define FIDX(q, cell, N)  ((cell) * NQ + (q))
#else
  #define FIDX(q, cell, N)  ((q) * (N) + (cell))
#endif

// kernel 本體完全共用
__global__ void lbm_step(const float* __restrict__ src, float* __restrict__ dst, ...)
{
    ...
    f[q] = src[FIDX(q, upstream_cell, N)];
    ...
    dst[FIDX(q, cell, N)] = f[q];
}
```

```bash
nvcc -O3 -arch=sm_86 -lineinfo -DUSE_AOS -o lbm_aos lbm.cu
nvcc -O3 -arch=sm_86 -lineinfo          -o lbm_soa lbm.cu
```

**實驗設計要點**：

1. **多個域大小各量一次**：128²、256²、512²、1024²、2048²。小域時整個陣列進得了 L2 cache，AoS 的劣勢會被藏起來；大域時才看得到真相。**這條曲線本身就是結論**——它會告訴你 L2 cache 幫了多少忙。
2. **每組跑 3 次取中位數**，不要取平均（單次的系統雜訊會污染平均）
3. **暖機 100 步不計時**，再跑 1000 步計時
4. **鎖定時脈**（如果有權限）：`sudo nvidia-smi -lgc <freq>`，或至少確認跑的時候溫度穩定
5. 輸出一張表：域大小 × 佈局 × MLUPS × 佔理論上限百分比

**預期結果**：

- 小域（128²）：兩者差距小，可能只有 1.2–1.5 倍
- 大域（1024² 以上）：SoA 快 2–5 倍
- SoA 在大域應達理論上限的 75–90%
- AoS 在大域可能只有 20–40%

如果你量出來 AoS 跟 SoA 幾乎一樣快，先別下結論——去 Nsight Compute 看 sectors per request，很可能是編譯器幫你做了什麼，或者你的實驗被別的瓶頸蓋住了。

#### 用 Nsight Compute 看 sector

```bash
ncu --section MemoryWorkloadAnalysis --section SpeedOfLight ./lbm_soa
```

要看的兩個東西：

- **`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`**：每個載入請求平均用了幾個 sector。完美 coalescing 的 f32 是 **4**。AoS 會接近 **32**。**這一個數字就是整節的結論。**
- **SpeedOfLight 區段的 Memory [%]**：DRAM 吞吐佔峰值的百分比。這對應你上面算的「佔理論上限」。

**版本注意**：Nsight Compute 的 metric 名稱和 section 名稱在不同版本間會改。用 `ncu --list-sections` 和 `ncu --query-metrics` 查你裝的那個版本實際有什麼，**不要照抄我寫的名字**。上面那串 metric 名稱在我印象中的 NCU 2022–2023 版是對的，但請自行驗證。

### 驗證方式

- AoS 版和 SoA 版的**數值結果必須完全一致**（同樣的加總順序，應該是 bit-exact）。這是實驗有效的前提，先驗這個
- SoA 的 sectors-per-request ≈ 4
- SoA 在大域達到理論上限 75% 以上
- 把 AoS/SoA 的 MLUPS 比值 對 域大小 畫成圖

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| AoS 跟 SoA 差不多快 | 域太小，全部進了 L2。加大到 1024² 以上 |
| SoA 只有理論上限的 50% | 檢查 `__restrict__` 有沒有加；檢查 block size（試 128、256）；檢查是不是 x 不在最連續維度 |
| 數字每次跑差 20% | 沒暖機，或 GPU 降頻 |
| `ncu` 跑超久 | 正常，profiler 會重放 kernel 很多次。用小域 profile，用大域量效能 |
| `ncu` 權限錯誤 | 需要 `--target-processes all` 或系統層級的 profiling 權限。Linux 上可能要設 `NVreg_RestrictProfilingToAdminUsers=0` |
| 不同域大小的 MLUPS 曲線很跳 | nx 不是 32 的倍數，導致每一列的起始位址沒對齊。用 padding 讓 pitch 是 32 的倍數（或用 `cudaMallocPitch`） |

**最後一點值得展開**：如果 `nx = 1000`，那麼第 1 列從 offset 1000 開始，不是 32 的倍數，於是每一列的 warp 都跨了一個 sector 邊界。把 `nx` padding 到 1024（或至少 32 的倍數），效能可能白得 10–20%。這是免費的。

---

1. **奇數步是同一個位址讀完再寫。** 這對 cache 很友善（讀進來的 line 還熱著），但也表示**讀跟寫之間有相依性**，編譯器沒辦法把所有讀提前。所以 in-place 的指令層級平行度（ILP）比雙陣列差一點。這是它速度提升有限的原因之一。
2. **奇偶步的 coalescing 不對稱。** 偶數步是完美的本地存取；奇數步有 8 個方向的鄰居偏移，其中 y 方向偏移會跨列。這跟雙陣列版的情況一樣，不是新問題。

#### 從外部讀取資料的注意事項

**AA-pattern 有個煩人的副作用：在奇數步結束時，記憶體裡的 DDF 是「反向存放」的。** 所以你的後處理（算 rho、u、輸出 VTK、傳回 Grasshopper）必須知道當前是哪個 parity。

實務做法：**只在偶數步結束後取資料**，或者寫一個 `parity`-aware 的 macroscopic kernel。前者簡單，建議先用。

### 具體練習

1. 先把 D2Q9 的 AA-pattern 在**紙上**用 D1Q3（一維三方向）推一遍，確認兩步之後等價於兩次 collide+stream。**這一步別跳，寫 code 前先確信它是對的，否則你會花好幾天 debug 一個你其實不理解的東西。**
2. 實作，週期邊界先（不含 bounce-back），驗證
3. 加入 bounce-back，驗證
4. 量記憶體佔用（`nvidia-smi` 或 `cudaMemGetInfo`），確認減半
5. 量 MLUPS，跟 L3 的雙陣列 SoA 版比
6. **找出你的卡在 in-place 下能跑的最大域，對比雙陣列下的最大域**

### 驗證方式

**最強的驗證：跑偶數步（例如 200 步），結果應與雙陣列版逐格一致。**

- f64 下：`max rel diff < 1e-13`
- f32 下：`max rel diff < 1e-5`（200 步累積）

**只跑偶數步很重要**，因為奇數步結束時記憶體佈局不同，沒辦法直接比。

另外：

- 記憶體佔用確實減半（`cudaMemGetInfo` 前後差）
- Lid-driven cavity 的穩態剖面與雙陣列版重合
- 總質量守恆同樣成立

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| 結果整個亂掉，但雙陣列版是對的 | 奇偶步的邏輯寫反了。試著把 `even` 條件反過來看是不是就對了 |
| 每兩步才對一次 | 你在奇數步結束時取資料。只在偶數步後比對 |
| 偶爾對偶爾錯，有隨機性 | 寫入位址有重疊 → 競爭條件。回去紙上推導 |
| 加了 `__restrict__` 之後結果變錯 | 就是這個原因。拿掉 |
| bounce-back 在 AA 下不知道怎麼寫 | 這是 AA-pattern 最容易出錯的地方。偶數步的 bounce-back 其實什麼都不用做（因為本來就寫回自己的 q̄）；奇數步才需要判斷。先只做週期邊界跑通，再加固體 |
| 速度完全沒變快 | 這是**正常的**。回去看本節開頭那三個真正的收益來源。你要量的是記憶體佔用和最大域 |
| 大域下反而變快很多 | 也正常——L2 cache 的有效容量加倍發揮作用了 |

**關於 bounce-back 在 AA 下的細節**：這在文獻上有幾種不同寫法，各有微妙差異（特別是固體節點附近的 parity 處理）。**如果卡住超過三天，一個合理的決定是：跳過 L4 的固體邊界，直接進 L5，之後在 L6 用 Esoteric Pull 取代 AA-pattern——Esoteric Pull 的 bounce-back 是隱式的，反而更簡單。** AA-pattern 的教學價值在於讓你理解「in-place streaming 是可能的」這件事，達到這個目的就夠了。

---

**（1）`M` 矩陣不是最常見的正交版本。** 教科書（d'Humières et al. 2002）的第 1 列是 `[-30, -11, ..., 8, ...]`，repo 用的是 `[-1, 0,...,0, 1,...,1]`。這是一個經過重新縮放的變體，本身可以是正確的（只要 `M⁻¹` 是用同一個 `M` 算出來的，而 repo 確實是用 `np.linalg.inv(M_np)`），但**它的矩不是正交歸一的，所以文獻上的 `s` 值不能直接套用**。移植時照抄 repo 的 `M`，不要去找論文的 `M` 換上去。

**（2）index 10 和 12 在 repo 裡用 s_v，但它們是 ghost。** 經典 MRT 通常給它們 `s_other`（強阻尼）。repo 給 `s_v`。這可能是刻意的變體，也可能不是。**影響：高 Re 下的穩定性。** 移植時照抄，但在 L7 之後可以當作一個可調參數實驗。

**（3）`self.tau_f = self.niu/3.0 + 0.5` 這一行，跟被註解掉的 `3.0*self.niu + 0.5` 不一樣。** 標準關係是：

```latex
\nu = c_s^2 \left(\tau - \tfrac{1}{2}\right) = \tfrac{1}{3}\left(\tau - \tfrac{1}{2}\right)
\quad\Longrightarrow\quad
\tau = 3\nu + \tfrac{1}{2}
```

也就是被註解掉的那一行才符合標準推導。**這會直接影響你算出來的實際 Reynolds 數。** 我不知道 repo 為什麼這樣寫（可能是它的 `niu` 變數有不同的縮放約定），**但你在移植時必須把這件事弄清楚**，否則你的 CUDA 版跟 taichi 版會「數值一致但物理不同」，L2 層級的驗證通過、物理驗證卻失敗。**用 Poiseuille 流的解析解去反推實際黏滯係數，是釐清這件事最直接的方法。**

#### MRT 在 CUDA 裡的實作：矩陣乘法要不要展開

三種寫法，效能差距可以到 2 倍：

**（A）真的做矩陣乘法（最慢）**

```cuda
float m[19] = {0};
for (int i = 0; i < 19; ++i)
    for (int j = 0; j < 19; ++j)
        m[i] += M[i][j] * f[j];        // 361 次 FMA，而且 M[i][j] 要從記憶體讀
```

**不要這樣寫。** 即使 `M` 在 `__constant__`，雙層迴圈的索引也會讓 `m[]` 和 `f[]` 掉到 local memory。

**（B）完全展開（`#pragma unroll` 或編譯期迴圈）**

```cuda
#pragma unroll
for (int i = 0; i < 19; ++i) {
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 19; ++j) s += M_CONST[i][j] * f[j];
    m[i] = s;
}
```

索引變編譯期常數，`m[]`、`f[]` 留在 register。編譯器**應該**會把乘以 0 的項消掉、把乘以 1 消成加法。實際上會不會，**看 `-Xptxas -v` 的指令數，不要相信我說的。**

**（C）手寫矩的和式（最快，也最容易寫錯）**

`M` 的元素只有 `{0, ±1, ±2}`，所以根本不需要乘法：

```cuda
// 依 repo 的 M 手動展開。以 m[3] = j_x 為例：
// M 第 3 列 = [0,1,-1,0,0,0,0, 1,-1,1,-1,1,-1,1,-1, 0,0,0,0]
const float jx = (f[1] - f[2])
               + (f[7] - f[8]) + (f[9] - f[10])
               + (f[11] - f[12]) + (f[13] - f[14]);

// m[0] = rho：全部相加
const float rho = f[0]+f[1]+f[2]+f[3]+f[4]+f[5]+f[6]+f[7]+f[8]+f[9]
                + f[10]+f[11]+f[12]+f[13]+f[14]+f[15]+f[16]+f[17]+f[18];

// m[13] = p_xy：M 第 13 列 = [0,...,0, 1,1,-1,-1, 0,0,0,0,0,0,0,0]（在 7..10 位置）
const float pxy = f[7] + f[8] - f[9] - f[10];
```

**（C）的正確作法：不要手抄。寫一個 Python 腳本從 repo 的 `M_np` 自動產生這段 CUDA 程式碼。**

```python
import numpy as np
# 從 repo 直接 import 或複製 M_np
names = [f'f[{j}]' for j in range(19)]
for i in range(19):
    terms = []
    for j in range(19):
        c = M_np[i, j]
        if c == 0:   continue
        if c == 1:   terms.append(f'+ {names[j]}')
        elif c == -1: terms.append(f'- {names[j]}')
        else:        terms.append(f'{"+" if c>0 else "-"} {abs(c)}.0f*{names[j]}')
    body = ' '.join(terms).lstrip('+ ')
    print(f'const float m{i} = {body};')
# inv_M 同理，產生反轉換
```

**這是這個里程碑最有效率的一步。** 它消除了手抄 361 個係數的出錯機會，而且當你想試別的 `M` 時改一行就好。

#### Register 壓力與 occupancy

D3Q19 MRT 的 register 需求：19 個 `f` + 19 個 `m` + ρ、u、暫存 ≈ **60–110 個 register**（實測看編譯器）。

Ampere / Ada 每個 SM 有 65536 個 32-bit register。能同時駐留的 thread 數：

```latex
\text{threads per SM} = \left\lfloor \frac{65536}{R_{\text{per thread}}} \right\rfloor
```

| register/thread | threads/SM | 佔滿載（1536 或 2048）的比例 |
| --- | --- | --- |
| 32 | 2048 | 100% |
| 64 | 1024 | 50–67% |
| 96 | 682 | 33–44% |
| 128 | 512 | 25–33% |
| 168 | 390 | 19–25% |
| 255（上限） | 257 | 13–17% |

看起來很糟，**但對 LBM 來說 25–50% 的 occupancy 通常就夠了**，原因是：

**Occupancy 的唯一目的是隱藏記憶體延遲。** 隱藏延遲有兩個來源：

- **TLP（thread-level parallelism）**：很多 warp 輪流跑 → 需要高 occupancy
- **MLP / ILP（memory-level / instruction-level parallelism）**：**同一個 thread 同時發出很多個互不相依的載入** → 不需要高 occupancy

LBM 的 pull kernel 一開頭就發出 **19 個彼此獨立的載入**。這是極高的 MLP，一個 thread 就能把記憶體管線填得很滿。所以低 occupancy 不痛。

**實務準則**：

- 不要為了 occupancy 犧牲 register。**寧可用 100 個 register 不 spill，也不要用 `-maxrregcount=64` 逼出 spill。** spill 到 local memory 是直接打在 DRAM 上，比低 occupancy 糟得多。
- 用 `__launch_bounds__(128)` 告訴編譯器你的 block size，它會據此分配 register 上限：

```cuda
__global__ void __launch_bounds__(128, 2)   // 128 threads/block，至少 2 blocks/SM
lbm_d3q19_mrt(...) { ... }
```

- **實測掃一次**：block size 用 64 / 128 / 256 / 512，`-maxrregcount` 用 不設 / 128 / 96 / 64，跑 16 組，畫成表。**最佳組合幾乎不可能靠推理猜到，一定要掃。** 這個掃描大概 2 小時，值得。
- 檢查 `-Xptxas -v` 的 `spill stores` 和 `spill loads`，**目標是 0**。

**降低 register 壓力的技巧**：

- 不要同時持有 `f[]` 和 `m[]`。碰撞前 `f` 可以直接就地覆寫成 `m`（如果你的手寫和式是小心排序的）。
- 或者：分段算。先算守恆量（ρ, j），存下來；再一組一組算剪應力矩。這會增加指令數但降低 peak register。
- 讓編譯器決定，通常它比你聰明。先不要手動最佳化，先量。

#### LES：為什麼只改剪應力那幾個

Smagorinsky 型的 LES（Large Eddy Simulation，大渦模擬）在 LBM 裡的作法是加上一個渦黏滯係數（eddy viscosity）：

```latex
\nu_{\text{total}} = \nu_0 + \nu_t,\qquad
\nu_t = (C_s \Delta)^2\, |\bar{S}|
```

而 `ν` 只透過一個管道進入 MRT——就是 `s_v`：

```latex
s_v = \frac{1}{\tau} = \frac{1}{3\nu_{\text{total}} + 0.5}
```

**所以加 LES 在程式上就是：每個 cell 算出自己的 `|S̄|`、算出自己的 `s_v`、只把它套到剪應力那幾個矩（9, 11, 13, 14, 15）上。其他 15 個鬆弛率一個字都不改。**

為什麼不能改別的？因為：

- **守恆矩（0, 3, 5, 7）**的鬆弛率必須是 0，否則質量/動量不守恆。改了就物理崩壞。
- **ghost modes（4, 6, 8, 16, 17, 18）**的鬆弛率是為了數值穩定而調的，跟物理黏滯係數無關。把湍流模型套上去等於讓數值阻尼隨流場變動，會引入非物理行為，而且可能不穩定。
- **只有剪應力矩對應到黏滯應力張量**。渦黏滯性的物理意義就是「未解析的小尺度渦對解析尺度的剪應力貢獻」，所以它只該出現在這裡。

`|S̄|` 怎麼算？好消息是 **MRT 已經把它交到你手上了**——非平衡的剪應力矩就是應變率張量：

```latex
\bar{S}_{\alpha\beta} \propto -\frac{s_v}{2\rho c_s^2}\left(m_{\alpha\beta} - m^{\text{eq}}_{\alpha\beta}\right)
```

也就是用 `m[9] - meq[9]`、`m[11] - meq[11]`、`m[13..15] - meq[13..15]` 這五個量組出 `|S̄|`。**這是 MRT 對 LES 的另一個結構性好處：不用額外的有限差分去算速度梯度，省掉鄰居存取。**

**但注意**：`s_v` 出現在 `|S̄|` 的式子裡，而 `s_v` 又取決於 `ν_t` 又取決於 `|S̄|`——這是隱式的。標準解法是解那個二次方程式得到閉式解（Hou et al. 1996 的作法），**具體式子請查該篇論文或 LBM 教科書（Krüger et al., "The Lattice Boltzmann Method", Springer 2017，第 17 章）**，我不憑記憶寫給你。

**L5 不要實作 LES。** 先把無 LES 的 MRT 驗證通過。LES 是 L5 之後的延伸，而且它會破壞你跟 taichi 版的逐格比對（除非 taichi 版也加）。

#### Kernel 骨架

```cuda
__global__ void __launch_bounds__(128)
lbm_d3q19_mrt(const float* __restrict__ src,
              float* __restrict__ dst,
              const uint8_t* __restrict__ solid,
              int nx, int ny, int nz,
              float s_v, float s_other)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const long long N    = (long long)nx * ny * nz;   // 注意：256^3 × 19 已接近 int 上限
    const long long cell = ((long long)z * ny + y) * nx + x;
    if (solid[cell]) return;

    // 1) Pull，含 bounce-back
    float f[19];
    #pragma unroll
    for (int q = 0; q < 19; ++q) {
        int xs = x - CX[q], ys = y - CY[q], zs = z - CZ[q];
        bool out = (xs<0)|(xs>=nx)|(ys<0)|(ys>=ny)|(zs<0)|(zs>=nz);
        long long nb = out ? cell : (((long long)zs*ny + ys)*nx + xs);
        bool bb = out || solid[nb];
        f[q] = src[(bb ? OPP[q] : q) * N + (bb ? cell : nb)];
    }

    // 2) 巨觀量
    float rho, ux, uy, uz;
    macroscopic(f, rho, ux, uy, uz);

    // 3) f → m（自動產生的手寫和式）
    float m[19];
    f_to_m(f, m);

    // 4) 鬆弛
    float meq[19];
    equilibrium_moments(rho, ux, uy, uz, meq);
    #pragma unroll
    for (int q = 0; q < 19; ++q) m[q] -= S[q] * (m[q] - meq[q]);

    // 5) m → f（自動產生）
    m_to_f(m, f);

    // 6) 寫回
    #pragma unroll
    for (int q = 0; q < 19; ++q) dst[q * N + cell] = f[q];
}
```

**索引型別的坑值得特別標出來**：`nx*ny*nz*19` 在 `int` 下，256³×19 = 3.19×10^8 還安全，但 384³×19 = 1.08×10^9 接近 `int` 上限 2.1×10^9，512³×19 = 2.55×10^9 **就溢位了**。溢位的症狀是隨機的記憶體毀損，極難 debug。**一開始就用 `long long` 或 `size_t`，成本幾乎為零。**

block 形狀建議 `dim3(64, 2, 1)` 或 `dim3(32, 4, 1)`——x 維度要夠寬才有 coalescing，但整體不要太大以免 register 撐不住。這也要掃。

### 具體練習

1. 寫 Python 產生器，從 repo 的 `M_np` 產出 `f_to_m` 和 `m_to_f` 的 CUDA 程式碼
2. 驗證產生器：在 Python 裡對隨機 `f` 做 `M @ f`，跟產生的式子逐項對照
3. 實作 kernel，週期邊界先，L2 層級單步 f64 比對
4. 加入 bounce-back，重驗
5. 釐清 `tau_f` 那個問題（用 Poiseuille 解析解）
6. 跑 lid-driven cavity 3D，用 repo 的 `geo_cavity.dat` 做幾何，跟 taichi 版比
7. 掃 block size × maxrregcount，畫效能表
8. 用 `ncu` 量 occupancy 和 DRAM throughput

### 驗證方式

- **單步 f64 逐格：`max rel diff < 1e-13`。** 這是最強的驗證，做到就幾乎確定 kernel 是對的
- 單步 f32：`max rel diff < 1e-6`
- 100 步 f32：`max rel diff < 1e-4`
- `-Xptxas -v`：`0 bytes spill`
- 3D lid-driven cavity 穩態速度場與 taichi 版 L2 誤差 < 1e-3
- Poiseuille 管流的拋物線剖面與解析解吻合，且反推出的 ν 與你設的一致
- MLUPS ≥ 理論上限的 70%

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| f64 下也差 1e-3 量級 | `M` 或 `M⁻¹` 的某幾列抄錯。用產生器，別手抄。或者 `meq` 的某項漏了 |
| 只有某些方向錯 | `CX/CY/CZ` 跟 repo 的 `e_f` 順序不一致，或 `OPP` 表錯 |
| 效能只有 30% 上限 | 看 `-Xptxas -v` 的 spill。有 spill 就先解 spill |
| 加了 `-maxrregcount` 反而變慢 | 逼出 spill 了。拿掉 |
| 大域跑出隨機亂數 | `int` 索引溢位。改 `long long` |
| occupancy 只有 25% 但效能很好 | 正常。LBM 靠 MLP 不靠 TLP。不要為了那個數字去最佳化 |
| 數值不穩定、幾百步後爆掉 | 先檢查 `s_other` 的公式；再檢查 τ 是否 > 0.5；再檢查最大速度 < 0.1（Mach 限制） |
| taichi 版穩定但 CUDA 版爆炸 | 十之八九是 index 10、12 的鬆弛率抄錯，或某個 ghost mode 的 `meq` 沒設 0 |
| 移植完發現 Re 跟預期差三倍 | 就是上面那個 `tau_f` 的問題 |

---

**判準建議**：對建築風環境這種以「穩態平均風場」為輸出的應用，FP16 的誤差通常遠小於邊界條件與湍流模型本身的不確定性。**如果 FP16 讓你能跑 2 倍大的域或 2 倍的解析度，那換來的物理準確度提升會遠大於 FP16 損失的數值精度。** 這是一個值得在報告裡明確論證的取捨。

#### Esoteric Pull：概念層次

Moritz Lehmann 提出（FluidX3D 的核心技術，論文：*Computation* 10(6), 92, 2022，"Esoteric Pull and Esoteric Push: Two Simple In-Place Streaming Schemes for the Lattice Boltzmann Method on GPUs"）。它同時解決三件事，這是它厲害的地方。

**（1）單陣列 in-place**

跟 AA-pattern 一樣是單陣列，但機制不同。AA-pattern 靠「奇偶步用不同的讀寫模式」，Esoteric Pull 靠「**每一對反向方向共用一個半步的位移**」：

概念上，對每一對 `(q, q̄)`，把其中一個方向的 DDF 存在「自己格子」，另一個存在「鄰居格子」。streaming 時，兩者剛好互換位置——讀進來、算完、寫回去時位置自然就對了。**每一步的讀寫模式都相同**，不需要奇偶交替。

**這比 AA-pattern 好在哪**：不需要 parity 狀態、後處理隨時可以做、程式碼路徑只有一條（沒有 `if (even)`），而且 warp 內不會因為 parity 發散。

**（2）隱式 bounce-back**

這是最漂亮的部分。因為每一對反向方向的儲存位置安排方式，**當上游是固體時，「讀鄰居的 q」和「讀自己的 q̄」在記憶體上退化成同一個操作**。

結果是：**bounce-back 不需要 `if` 判斷，不需要讀 solid mask 決定走哪條路**。只要固體格點的 DDF 保持不動（不更新），邊界條件就自動正確了。

對你的案例的意義：

- **消除了 streaming 迴圈裡 19 個分支**，也就消除了 warp divergence
- **省掉每步讀 solid mask 的頻寬**（雖然只有 1 byte/cell，但在 38 bytes 的預算裡也佔 2.6%）
- 建築幾何複雜、固體佔比高時，收益更大

**（3）完美 coalescing**

Esoteric Pull 的記憶體存取模式設計成：一個 warp 的 32 個 thread 對每一個 `q` 的存取都是連續的。**沒有任何一個方向會退化成跨步存取。** 這跟樸素的 pull 不同——樸素 pull 在 y、z 方向的偏移會讓 warp 讀到不同的列，雖然仍然合併但會跨 cache line。

#### 誠實的範圍標註

**我對 Esoteric Pull 的實作細節（確切的索引公式、哪些方向存在哪裡）沒有足夠把握寫出可直接用的程式碼。** 這份文件只給到概念層次，是刻意的。

要實作，去這兩個來源，順序建議：

1. **FluidX3D 的原始碼**（GitHub: `ProjectPhysX/FluidX3D`），特別是 OpenCL kernel 裡的 `load_f` / `store_f` 函式和 `EQUILIBRIUM_BOUNDARIES` 相關部分。**這是最可靠的參考，因為它是能跑的程式碼。**
2. **Esoteric Pull 那篇論文**（*Computation* 2022, 10, 92），有索引推導的圖。

FluidX3D 用 OpenCL 不是 CUDA，但兩者的 kernel 語言幾乎可以一對一翻譯（`get_global_id(0)` → `blockIdx.x*blockDim.x+threadIdx.x`，`__global` → 指標，`barrier()` → `__syncthreads()`）。

**另外一個誠實的建議：Esoteric Pull 的實作難度明顯高於前面所有里程碑。如果 L6 的前半段（FP16）已經證明了 2 倍的收益、而你的時間預算吃緊，「只做 FP16 + 沿用 L4 的 AA-pattern」是完全合理的停損點。** Esoteric Pull 相對 AA-pattern 的額外收益主要是隱式 bounce-back 與更好的 coalescing，量級大概在 10–30%，不是另一個 2 倍。

### 具體練習

1. 寫一個小程式，把你的 LBM 跑到穩態的 DDF 場，逐個轉成 FP16 再轉回來，量最大相對誤差。**確認它真的是 ~5×10⁻⁴ 而不是更糟**（如果更糟，代表你的 DDF 有離群值，要查）
2. 把 L5 的 kernel 改成 FP16 儲存，跑通
3. 跑上面四項精度驗證
4. 量 MLUPS 和記憶體佔用，跟 L5 的 FP32 版比。**這個數字是整條學習軌最重要的輸出之一**
5. 試「只存非平衡部分」的變體，看精度改善多少
6. 試 `__half2` 向量化載入，看效能改善多少
7. （選做）讀 FluidX3D 原始碼，寫一份 Esoteric Pull 的索引推導筆記

### 驗證方式

- FP16 round-trip 相對誤差 ≈ 5×10⁻⁴
- 100 步後速度場與 FP32 版相對 L2 誤差 < 1e-3
- Cavity 穩態剖面誤差與 FP32 版同數量級
- 10⁵ 步不發散、質量漂移率可接受
- MLUPS 相對 FP32 版有**明顯**提升（預期 1.6–2.0 倍；若只有 1.1 倍，代表你還有別的瓶頸，回去 profile）
- 記憶體佔用降為 FP32 單陣列的一半、FP32 雙陣列的四分之一

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| FP16 版立刻發散 | DDF 有值落在 FP16 範圍外，或初始化時用了 0 造成下溢。印出 `f.min()/f.max()` 看 |
| 精度比預期差 10 倍 | 就是非平衡部分被淹沒的問題。改存 `f - w` |
| 速度只快 1.1 倍 | 沒到 bandwidth bound，可能被 register spill 或 solid mask 的讀取卡住。`ncu` 看 DRAM throughput |
| `__half2float` 在 host code 編譯失敗 | 那些是 device-only 函式。host 端轉換用 `__float2half` 的 host 版本或 numpy 的 `float16` |
| CuPy 傳 `__half` 陣列進 kernel 出錯 | CuPy 的 `cp.float16` 對應 `__half`，但 `RawKernel` 的型別檢查有時需要用 `cp.uint16` 繞過再在 kernel 裡 `reinterpret_cast`。**這個細節我不確定各版本行為，查 CuPy 文件的 "Custom kernels" 一節** |
| 長時程緩慢漂移 | FP16 的截斷偏差。考慮 stochastic rounding（但 CUDA 沒有內建，要自己做，複雜度高） |
| Esoteric Pull 怎麼寫都不對 | 正常。這是整份文件最難的部分。回去讀 FluidX3D 原始碼，不要自己推 |

---

```python
import taichi as ti
ti.init(arch=ti.cuda, kernel_profiler=True, print_ir=False)

# ... 跑模擬 ...
ti.sync()
ti.profiler.print_kernel_profiler_info()   # 較新版本
# 較舊版本是 ti.print_kernel_profile_info()
```

**版本注意**：taichi 的 profiler API 名稱在 1.0 前後改過（`ti.print_kernel_profile_info` → `ti.profiler.print_kernel_profiler_info`），**請查你裝的版本的文件「Profiler」一節，不要照抄。** 另外 `ti.profiler.clear_kernel_profiler_info()` 可以在暖機後清掉統計，這樣印出來的就只有正式量測那段。

它會印出一張每個 kernel 的耗時表，長這樣：

```
[ 45.32%] colission            min  1.234 ms   avg  1.312 ms   max  1.501 ms   total  1.312 s
[ 30.11%] streaming1           min  0.812 ms   avg  0.871 ms   max  0.990 ms   total  0.871 s
...
```

**這張表決定「只換最熱的 kernel」這個結局可不可行。** 如果 `colission` + `streaming` 合計佔 90%，其餘都是零頭，那你只要把那兩個換成 CUDA 就拿到幾乎全部的收益，邊界條件、後處理、VTK 輸出可以繼續用 taichi 寫（開發快、好維護）。這是最務實的結局，也是最常見的最佳解。

repo 的 `Single_phase/example_cavity.py` 第 4 行已經有這兩個旗標的位置：

```python
ti.init(arch=ti.cpu, dynamic_index=False, kernel_profiler=False, print_ir=False)
```

把 `arch` 改成 `ti.cuda`、`kernel_profiler` 改成 `True` 就能開始量。

### 怎麼判斷「已經接近頻寬上限」

三重確認，全部通過才算數：

**（1）算術上的確認**

```latex
\text{有效頻寬} = \frac{n_x n_y n_z \times B_{\text{per cell}} \times N_{\text{steps}}}{t}
```

把它除以 `bandwidthTest` 量到的 Device-to-Device 頻寬。**> 80% 就是接近上限。**

**（2）Profiler 的確認**：Nsight Compute 的 "Memory [%]" 應該給出接近的數字。**兩個數字對不起來的話，代表你的 `B_per_cell` 算錯了**（常見原因：忘了算 solid mask、或 cache 把某些存取吃掉了讓實際 DRAM 流量低於理論值）。

**（3）敏感度測試**：把碰撞模型從 MRT 換成 BGK（算術量降到約 1/5）。**如果 MLUPS 幾乎不變，就證明了算術不是瓶頸。** 這是最直接、最有說服力的實驗，五分鐘就能做，**強烈建議做，因為它是你所有論證的基石**。

### 三種結局的決策標準

設 `R = CUDA 最佳版 MLUPS / taichi 最佳版 MLUPS`（注意分母是**公平比較過的** taichi 版，見上面的清單）。

#### 結局 A：留著 taichi

**條件**：`R < 1.3`，且 taichi 版已達理論上限 70% 以上。

**理由**：30% 的效能換不來以下這些：跨平台（CPU / CUDA / Vulkan / Metal 同一份程式碼）、跟 numpy/Rhino 的無痛整合、開發迭代速度（改一行立刻跑，不用編譯）、除錯容易度、給後續接手的人（建築系研究生）的可讀性。

**對主線的意義**：把這三個月學到的東西**回頭改善 taichi 版**——kernel 融合、SoA 佈局、block size 調整（taichi 有 `ti.loop_config(block_dim=...)`）。你很可能光靠這些就在 taichi 裡拿到 2 倍。**這是最常見、也常常是最好的結局。**

#### 結局 B：只換最熱的 kernel

**條件**：`1.3 < R < 2.5`，且 taichi profiler 顯示 1–2 個 kernel 佔 85% 以上時間。

**做法**：那 1–2 個 kernel 寫成 CUDA，編成 .so 或 CuPy RawKernel，其餘留在 taichi。記憶體用 CuPy 陣列當共用載體（taichi 有 `ti.field.from_numpy`/`to_numpy`，但更好的是用 `ti.ndarray` 或 external array 介面直接共用 device pointer，**避免每步 host-device 來回**）。

**風險**：兩套系統的記憶體佈局要對齊，介面層本身是 bug 溫床。**在做之前先量一次「介面本身的 overhead」**：寫一個什麼都不做的 CUDA kernel，從 taichi 的迴圈裡呼叫它，量 launch overhead。如果 overhead 佔了單步時間的 5% 以上，這個方案的收益就被吃掉了。

#### 結局 C：整包換成 CUDA

**條件**：`R > 2.5`，**或者**你需要的功能 taichi 做不到：

- FP16 儲存（taichi 有 `ti.f16`，**但我不確定它在 CUDA backend 上的成熟度與效能，這需要你自己實測**——如果 taichi 的 f16 能用且快，結局 C 的理由就消失一半）
- Esoteric Pull 那種精細的記憶體佈局控制
- Multi-GPU domain decomposition
- 跟 C++ 生態（VTK、CGAL、幾何處理）的直接整合

**代價要誠實列出**：開發時間增加 3–5 倍、失去 CPU fallback（沒有 NVIDIA 卡的使用者完全不能跑，**這對一個要發給建築師用的 Grasshopper 工具是嚴重問題**）、部署複雜度（要編譯、要對 CUDA 版本）、維護負擔。

**對 Grasshopper 主線特別重要的一點**：如果你的工具要給別人用，CUDA-only 表示使用者必須有 NVIDIA 顯卡。建築事務所的機器有 Quadro/RTX 的比例不低，但 Mac 使用者直接出局。taichi 的 Vulkan/Metal backend 能涵蓋這些。**這個相容性考量可能比效能數字更決定性。**

### 具體練習

1. 建立可重現的 benchmark 腳本（兩邊共用同一個驅動框架、同一個計時邏輯）
2. 把 taichi 版改成融合 + SoA（這本身可能就是主線的一個貢獻）
3. 填完四列比較表，域大小掃 128³、192³、256³
4. 用 `ncu` profile CUDA 版與 taichi 版，比對 DRAM throughput
5. 做 MRT → BGK 的敏感度測試
6. 量 CUDA kernel 從 taichi 迴圈呼叫的 launch overhead
7. **寫出決策，並寫出支持它的三個數字**

### 預期會踩的坑

| 症狀 | 原因 |
| --- | --- |
| taichi 第一次量特別慢 | JIT。加暖機 |
| taichi 第二次跑比第一次快很多 | offline cache。確保每次量測前狀態一致 |
| CUDA 版「快 5 倍」 | 幾乎一定是比較不公平。回去看公平性清單，特別是 kernel 融合那項 |
| 兩邊的 MLUPS 都遠低於理論上限 | 域太小，kernel launch overhead 佔比高。加大域，或增加每次計時的步數 |
| `ncu` 對 taichi 產生的 kernel 給不出名字 | 正常，taichi 的 kernel 名稱是混淆過的。用 `--print-summary per-kernel` 至少能看到分佈 |
| 結果每天不一樣 | 環境。鎖時脈、關背景程序、確認沒有別人在用同一張卡 |
| 決策做不出來，數字在灰色地帶 | 那答案就是結局 A 或 B。**灰色地帶不是換語言的理由。** |

---
