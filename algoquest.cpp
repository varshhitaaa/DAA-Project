// =====================================================================
//  AlgoQuest - The Algorithmic Dungeon  (C++ port)
//  A console-mode port of the 6-room dungeon from the web version.
// 
//  Algorithms implemented (one per room):
//    Room 1  BFS (shortest path) + DFS (reachability)
//    Room 2  0/1 Knapsack - Dynamic Programming
//    Room 3  Kruskal's MST - Greedy + DSU (union-find, path compression)
//    Room 4  Minimax with Alpha-Beta Pruning - Game theory
//    Room 5  Quick Sort - Divide & Conquer
//    Room 6  N-Queens - Backtracking
//
//  Leaderboard uses a std::priority_queue (max-heap by score).
//
//  Build:   g++ -std=c++17 -O2 algoquest.cpp -o algoquest
//  Run:     ./algoquest
// =====================================================================

#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <climits>
#include <sstream>
#include <iomanip>

using namespace std;

// --------- ANSI colors (degrade gracefully on terminals without them) ---
namespace C {
    const string RESET  = "\033[0m";
    const string BOLD   = "\033[1m";
    const string DIM    = "\033[2m";
    const string RED    = "\033[31m";
    const string GREEN  = "\033[32m";
    const string YELLOW = "\033[33m";
    const string BLUE   = "\033[34m";
    const string MAG    = "\033[35m";
    const string CYAN   = "\033[36m";
    const string WHITE  = "\033[37m";
}

// --------- Hero / global game state ------------------------------------
struct Hero {
    string name = "Hero";
    int hp = 100, maxHp = 100;
    int gold = 0, score = 0, rooms = 0;
    vector<bool> cleared = vector<bool>(6, false);
};

Hero hero;

struct LBEntry {
    string name;
    int    score;
    int    rooms;
    bool operator<(const LBEntry& o) const { return score < o.score; } // max-heap
};

vector<LBEntry> leaderboard = {
    {"Gandalf",  920, 6},
    {"Lancelot", 780, 6},
    {"Merlin",   650, 5},
    {"Arthur",   540, 5},
    {"Frodo",    430, 4},
};

void pauseMsg(const string& s = "\nPress ENTER to continue...") {
    cout << C::DIM << s << C::RESET;
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void showStats() {
    cout << "\n" << C::BOLD << "  ["
         << "Hero: " << hero.name
         << " | HP: " << C::RED << hero.hp << "/" << hero.maxHp << C::RESET << C::BOLD
         << " | Gold: " << C::YELLOW << hero.gold << C::RESET << C::BOLD
         << " | Score: " << C::GREEN << hero.score << C::RESET << C::BOLD
         << " | Rooms: " << C::CYAN << hero.rooms << "/6" << C::RESET << C::BOLD
         << "]" << C::RESET << "\n";
}

void reward(int score, int gold, const string& item = "") {
    hero.score += max(0, score);
    hero.gold  += gold;
    cout << "\n" << C::GREEN << "+ " << score << " score   "
         << C::YELLOW << "+ " << gold << " gold" << C::RESET;
    if (!item.empty()) cout << "   " << C::CYAN << "Item: " << item << C::RESET;
    cout << "\n";
}

void completeRoom(int idx, int score, int gold, const string& item = "") {
    if (!hero.cleared[idx]) {
        hero.cleared[idx] = true;
        hero.rooms++;
    }
    reward(score, gold, item);
}

// =====================================================================
// ROOM 1 - The Haunted Maze  (BFS + DFS)
// =====================================================================
struct MazeRoom {
    vector<string> layout = {
        "############",
        "#P..#......#",
        "#.#.#.####.#",
        "#.#...#....#",
        "#.###.#.##.#",
        "#...#.#.#..#",
        "###.#.#.#.##",
        "#.........E#",
        "############"
    };
    vector<string> grid;
    int R, Cn, pr, pc, er, ec;
    int moves = 0;
    int optimalLen = 0;
    vector<pair<int,int>> optimalPath;
    bool showHint = false;

    void init() {
        grid = layout;
        R = (int)grid.size(); Cn = (int)grid[0].size();
        moves = 0; showHint = false;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < Cn; j++) {
                if (grid[i][j] == 'P') { pr = i; pc = j; grid[i][j] = '.'; }
                if (grid[i][j] == 'E') { er = i; ec = j; }
            }
        optimalPath = bfs();
        optimalLen  = (int)optimalPath.size() > 0 ? (int)optimalPath.size() - 1 : 0;
    }

    // BFS - finds the shortest path using a queue. O(V+E).
    vector<pair<int,int>> bfs() {
        vector<vector<bool>>  vis(R, vector<bool>(Cn, false));
        vector<vector<pair<int,int>>> par(R, vector<pair<int,int>>(Cn, {-1,-1}));
        queue<pair<int,int>> q;
        q.push({pr, pc});
        vis[pr][pc] = true;
        int dr[] = {-1,1,0,0}, dc[] = {0,0,-1,1};
        bool found = (pr == er && pc == ec);
        while (!q.empty() && !found) {
            auto [r, c] = q.front(); q.pop();
            for (int k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= Cn) continue;
                if (vis[nr][nc] || grid[nr][nc] == '#') continue;
                vis[nr][nc] = true;
                par[nr][nc] = {r, c};
                if (nr == er && nc == ec) { found = true; break; }
                q.push({nr, nc});
            }
        }
        vector<pair<int,int>> path;
        if (!found) return path;          // unreachable: empty path
        pair<int,int> cur = {er, ec};
        while (cur.first != -1) {
            path.push_back(cur);
            cur = par[cur.first][cur.second];
        }
        reverse(path.begin(), path.end());
        return path;
    }

    // DFS - finds ANY path (reachability). Uses an explicit stack. O(V+E).
    vector<pair<int,int>> dfs() {
        vector<vector<bool>>  vis(R, vector<bool>(Cn, false));
        stack<pair<int,int>> st;
        st.push({pr, pc});
        vector<pair<int,int>> order;
        int dr[] = {-1,1,0,0}, dc[] = {0,0,-1,1};
        while (!st.empty()) {
            auto [r, c] = st.top(); st.pop();
            if (vis[r][c]) continue;
            vis[r][c] = true;
            order.push_back({r, c});
            if (r == er && c == ec) break;
            for (int k = 3; k >= 0; k--) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= Cn) continue;
                if (vis[nr][nc] || grid[nr][nc] == '#') continue;
                st.push({nr, nc});
            }
        }
        return order;
    }

    void draw() {
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < Cn; j++) {
                if (i == pr && j == pc)      cout << C::YELLOW << "P" << C::RESET;
                else if (i == er && j == ec) cout << C::GREEN  << "E" << C::RESET;
                else if (grid[i][j] == '#')  cout << C::DIM    << "#" << C::RESET;
                else {
                    bool onPath = false;
                    if (showHint) {
                        for (auto& p : optimalPath)
                            if (p.first == i && p.second == j) { onPath = true; break; }
                    }
                    if (onPath) cout << C::CYAN << "." << C::RESET;
                    else        cout << " ";
                }
            }
            cout << "\n";
        }
    }

    bool step(int dr, int dc) {
        int nr = pr + dr, nc = pc + dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= Cn) return false;
        if (grid[nr][nc] == '#') {
            hero.hp = max(0, hero.hp - 2);
            cout << C::RED << "*thud* You bumped a wall! -2 HP\n" << C::RESET;
            return false;
        }
        pr = nr; pc = nc; moves++;
        return true;
    }

    void play() {
        init();
        cout << "\n" << C::BOLD << "== Room 1: The Haunted Maze ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: BFS (shortest path) / DFS (reachability)\n"
             << "Data Structures: Graph (2D grid), Queue (BFS), Stack (DFS)\n"
             << "Time Complexity: O(V + E)\n" << C::RESET;
        cout << "\nNavigate to " << C::GREEN << "E" << C::RESET
             << ". Controls: w/a/s/d = move, h = toggle BFS hint (-15), "
             << "f = show DFS visit count, q = leave\n";

        while (true) {
            cout << "\n";
            draw();
            cout << "\nMoves: " << C::BOLD << moves << C::RESET
                 << "   Optimal (BFS): " << C::BOLD << optimalLen << C::RESET;
            if (optimalLen == 0) cout << C::RED << "  (exit unreachable!)" << C::RESET;
            cout << "\n> ";

            char cmd; cin >> cmd;
            bool moved = false;
            if (cmd == 'w' || cmd == 'W') moved = step(-1, 0);
            else if (cmd == 's' || cmd == 'S') moved = step(1, 0);
            else if (cmd == 'a' || cmd == 'A') moved = step(0, -1);
            else if (cmd == 'd' || cmd == 'D') moved = step(0, 1);
            else if (cmd == 'h' || cmd == 'H') { showHint = !showHint;
                cout << C::CYAN << (showHint ? "BFS hint ON (-15 to score)\n"
                                             : "BFS hint OFF\n") << C::RESET; }
            else if (cmd == 'f' || cmd == 'F') {
                auto ord = dfs();
                cout << C::MAG << "DFS visited " << ord.size()
                     << " cells before reaching E (BFS visited "
                     << optimalLen + 1 << " on its path)\n" << C::RESET;
            }
            else if (cmd == 'q' || cmd == 'Q') return;
            else cout << "Unknown command.\n";

            if (moved && pr == er && pc == ec) {
                int extra = moves - optimalLen;
                int bonus = max(0, 50 - extra * 3);
                int score = max(0, 100 + bonus - (showHint ? 15 : 0));
                cout << C::GREEN << "\n** You reached the exit! **\n" << C::RESET;
                cout << "Your moves: " << moves
                     << "   BFS optimal: " << optimalLen
                     << "   Efficiency bonus: +" << bonus << "\n";
                completeRoom(0, score, 30, "Ancient Map");
                pauseMsg();
                return;
            }
        }
    }
};

// =====================================================================
// ROOM 2 - The Treasure Vault  (0/1 Knapsack via DP)
// =====================================================================
struct TreasureRoom {
    struct Item { string name; int value, weight; };
    vector<Item> items = {
        {"Golden Crown",   80, 7},
        {"Ruby Necklace",  60, 4},
        {"Silver Sword",   40, 3},
        {"Ancient Tome",   30, 2},
        {"Diamond Ring",   50, 1},
        {"Jade Statue",    70, 6},
        {"Pearl Chalice",  45, 3},
        {"Mystic Orb",     55, 4},
    };
    int capacity = 12;
    vector<bool> picked;
    int optimalValue = 0;
    bool showHint = false;

    // 0/1 Knapsack via bottom-up DP: dp[i][w]. O(n*W) time and space.
    int solveDP() {
        int n = (int)items.size(), W = capacity;
        vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
        for (int i = 1; i <= n; i++)
            for (int w = 0; w <= W; w++) {
                dp[i][w] = dp[i - 1][w];
                if (items[i - 1].weight <= w)
                    dp[i][w] = max(dp[i][w],
                                   dp[i - 1][w - items[i - 1].weight] + items[i - 1].value);
            }
        return dp[n][W];
    }

    int curWeight() const {
        int w = 0;
        for (size_t i = 0; i < items.size(); i++) if (picked[i]) w += items[i].weight;
        return w;
    }
    int curValue() const {
        int v = 0;
        for (size_t i = 0; i < items.size(); i++) if (picked[i]) v += items[i].value;
        return v;
    }

    void draw() {
        cout << "\nCapacity: " << curWeight() << "/" << capacity
             << " kg   Value: " << curValue() << " gold\n";
        for (size_t i = 0; i < items.size(); i++) {
            cout << "  " << (picked[i] ? C::GREEN + string("[x]") + C::RESET : "[ ]")
                 << " " << setw(2) << i + 1 << ". "
                 << setw(16) << left << items[i].name
                 << " value=" << setw(3) << items[i].value
                 << "  weight=" << items[i].weight << "\n";
        }
        if (showHint)
            cout << C::CYAN << "DP optimal value: " << optimalValue
                 << " gold\n" << C::RESET;
    }

    void play() {
        picked.assign(items.size(), false);
        showHint = false;
        optimalValue = solveDP();

        cout << "\n" << C::BOLD << "== Room 2: The Treasure Vault ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: 0/1 Knapsack (Dynamic Programming)\n"
             << "Data Structure: 2D DP table (n x W)\n"
             << "Time Complexity: O(n * W) = O("
             << items.size() * capacity << ")\n" << C::RESET;
        cout << "\nPick items to maximize value under a weight cap of "
             << capacity << " kg. Commands: 1-8 toggle item, "
             << "h hint (-15), g go (finish), q leave\n";

        while (true) {
            draw();
            cout << "> ";
            string cmd; cin >> cmd;
            if (cmd == "q" || cmd == "Q") return;
            if (cmd == "h" || cmd == "H") { showHint = !showHint; continue; }
            if (cmd == "g" || cmd == "G") {
                int my = curValue();
                double ratio = optimalValue ? (double)my / optimalValue : 1.0;
                int score = max(0, (int)floor(150 * ratio) - (showHint ? 15 : 0));
                bool perfect = (my == optimalValue);
                cout << (perfect ? C::GREEN + string("** PERFECT DP SOLUTION! **")
                                 : string("Vault cleared."))
                     << C::RESET << "\n";
                cout << "Your haul: " << my << " gold   DP optimal: "
                     << optimalValue << " gold   Efficiency: "
                     << (int)(ratio * 100) << "%\n";
                completeRoom(1, score, my, perfect ? "DP Master Token" : "");
                pauseMsg();
                return;
            }
            try {
                int k = stoi(cmd);
                if (k >= 1 && k <= (int)items.size()) {
                    int idx = k - 1;
                    if (!picked[idx] && curWeight() + items[idx].weight > capacity) {
                        cout << C::RED << "Too heavy!\n" << C::RESET;
                    } else picked[idx] = !picked[idx];
                }
            } catch (...) { cout << "Unknown command.\n"; }
        }
    }
};

// =====================================================================
// ROOM 3 - Bridge of Merchants  (Kruskal's MST with DSU)
// =====================================================================
struct BridgeRoom {
    struct Edge { int u, v, cost, idx; };
    vector<string> islands = {
        "Emerald Isle", "Ruby Coast", "Sapphire Point",
        "Gold Harbor",  "Silver Bay", "Obsidian Rock"
    };
    vector<Edge> edges = {
        {0,1,12,0},{0,2,25,1},{0,3,20,2},
        {1,2,8, 3},{1,4,15,4},{2,3,18,5},
        {2,4,30,6},{3,4,10,7},{3,5,22,8},
        {4,5,14,9},{1,5,35,10},{0,5,40,11}
    };
    vector<bool> chosen;
    int optimalCost = 0;
    bool showHint = false;

    // DSU with path compression + union by implicit order. Near O(α(n)).
    struct DSU {
        vector<int> p;
        DSU(int n) : p(n) { for (int i = 0; i < n; i++) p[i] = i; }
        int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
        bool unite(int a, int b) {
            int ra = find(a), rb = find(b);
            if (ra == rb) return false;
            p[rb] = ra;
            return true;
        }
    };

    // Kruskal's MST: sort edges by cost, union with DSU, skip cycles.
    // O(E log E).
    int kruskal() {
        vector<Edge> es = edges;
        sort(es.begin(), es.end(),
             [](const Edge& a, const Edge& b){ return a.cost < b.cost; });
        DSU d((int)islands.size());
        int cost = 0, taken = 0;
        for (auto& e : es) {
            if (d.unite(e.u, e.v)) {
                cost += e.cost;
                if (++taken == (int)islands.size() - 1) break;
            }
        }
        return cost;
    }

    bool isConnected() {
        DSU d((int)islands.size());
        for (size_t i = 0; i < edges.size(); i++)
            if (chosen[i]) d.unite(edges[i].u, edges[i].v);
        int root = d.find(0);
        for (size_t i = 1; i < islands.size(); i++)
            if (d.find((int)i) != root) return false;
        return true;
    }

    int myCost() {
        int s = 0;
        for (size_t i = 0; i < edges.size(); i++) if (chosen[i]) s += edges[i].cost;
        return s;
    }

    void draw() {
        cout << "\nIslands:\n";
        for (size_t i = 0; i < islands.size(); i++)
            cout << "  [" << i << "] " << islands[i] << "\n";
        cout << "\nEdges (toggle by number):\n";
        for (size_t i = 0; i < edges.size(); i++) {
            cout << "  " << setw(2) << i + 1 << ". "
                 << (chosen[i] ? C::GREEN + string("[x]") + C::RESET : "[ ]")
                 << " " << islands[edges[i].u] << " <-> " << islands[edges[i].v]
                 << "  cost " << edges[i].cost << "\n";
        }
        cout << "\nMy cost: " << myCost()
             << "   Connected: " << (isConnected() ? "YES" : "NO");
        if (showHint) cout << C::CYAN << "   Kruskal optimal: "
                           << optimalCost << C::RESET;
        cout << "\n";
    }

    void play() {
        chosen.assign(edges.size(), false);
        showHint = false;
        optimalCost = kruskal();

        cout << "\n" << C::BOLD << "== Room 3: Bridge of Merchants ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: Kruskal's MST (Greedy)\n"
             << "Data Structures: Edge list, DSU with path compression\n"
             << "Time Complexity: O(E log E)\n" << C::RESET;
        cout << "\nConnect ALL islands with minimum total cost.\n"
             << "Commands: 1-" << edges.size() << " toggle edge, "
             << "h hint (-15), g build (finish), q leave\n";

        while (true) {
            draw();
            cout << "> ";
            string cmd; cin >> cmd;
            if (cmd == "q" || cmd == "Q") return;
            if (cmd == "h" || cmd == "H") { showHint = !showHint; continue; }
            if (cmd == "g" || cmd == "G") {
                if (!isConnected()) {
                    cout << C::RED << "All islands must be connected!\n" << C::RESET;
                    continue;
                }
                int my = myCost();
                bool perfect = (my == optimalCost);
                double ratio = my ? (double)optimalCost / my : 1.0;
                int score = max(0, (int)floor(200 * ratio) - (showHint ? 15 : 0));
                cout << (perfect ? C::GREEN + string("** PERFECT MST! **")
                                 : string("Islands connected."))
                     << C::RESET << "\n"
                     << "Your cost: " << my << "   Kruskal optimal: "
                     << optimalCost << "\n";
                completeRoom(2, score, 50, perfect ? "Merchant's Seal" : "");
                pauseMsg();
                return;
            }
            try {
                int k = stoi(cmd);
                if (k >= 1 && k <= (int)edges.size()) chosen[k - 1] = !chosen[k - 1];
            } catch (...) { cout << "Unknown command.\n"; }
        }
    }
};

// =====================================================================
// ROOM 4 - Dark Lord's Chessboard  (Minimax + Alpha-Beta, 4x4 4-in-a-row)
// =====================================================================
struct ChessRoom {
    static const int N = 4;
    char b[N][N];
    bool over = false;

    void reset() {
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) b[i][j] = '.';
        over = false;
    }

    // Returns 'X', 'O', or '\0' (no winner yet).
    char winner(char g[N][N]) {
        for (int i = 0; i < N; i++) {
            if (g[i][0] != '.') {
                bool all = true;
                for (int j = 1; j < N; j++) if (g[i][j] != g[i][0]) { all = false; break; }
                if (all) return g[i][0];
            }
            if (g[0][i] != '.') {
                bool all = true;
                for (int j = 1; j < N; j++) if (g[j][i] != g[0][i]) { all = false; break; }
                if (all) return g[0][i];
            }
        }
        if (g[0][0] != '.') {
            bool all = true;
            for (int i = 1; i < N; i++) if (g[i][i] != g[0][0]) { all = false; break; }
            if (all) return g[0][0];
        }
        if (g[0][N - 1] != '.') {
            bool all = true;
            for (int i = 1; i < N; i++) if (g[i][N - 1 - i] != g[0][N - 1]) { all = false; break; }
            if (all) return g[0][N - 1];
        }
        return '\0';
    }

    bool isFull(char g[N][N]) {
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (g[i][j] == '.') return false;
        return true;
    }

    // Heuristic: sum of line scores. AI = 'O' (maximizer), player = 'X' (minimizer).
    int evaluate(char g[N][N]) {
        auto lineScore = [](const vector<char>& line) {
            int ai = 0, pl = 0;
            for (char c : line) { if (c == 'O') ai++; else if (c == 'X') pl++; }
            if (ai > 0 && pl > 0) return 0;
            if (ai > 0) return (int)pow(10, ai - 1);
            if (pl > 0) return -(int)pow(10, pl - 1);
            return 0;
        };
        int s = 0;
        for (int i = 0; i < N; i++) {
            vector<char> row(N), col(N);
            for (int j = 0; j < N; j++) { row[j] = g[i][j]; col[j] = g[j][i]; }
            s += lineScore(row);
            s += lineScore(col);
        }
        vector<char> d1(N), d2(N);
        for (int i = 0; i < N; i++) { d1[i] = g[i][i]; d2[i] = g[i][N - 1 - i]; }
        s += lineScore(d1);
        s += lineScore(d2);
        return s;
    }

    // Minimax with alpha-beta pruning. Best cut: worst-case O(b^d),
    // with ordering ~O(b^(d/2)).
    int minimax(char g[N][N], int depth, bool isMax, int alpha, int beta) {
        char w = winner(g);
        if (w == 'O') return 1000 - depth;
        if (w == 'X') return -1000 + depth;
        if (isFull(g) || depth >= 4) return evaluate(g);
        if (isMax) {
            int best = INT_MIN;
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
                if (g[i][j] == '.') {
                    g[i][j] = 'O';
                    best = max(best, minimax(g, depth + 1, false, alpha, beta));
                    g[i][j] = '.';
                    alpha = max(alpha, best);
                    if (beta <= alpha) return best;  // prune
                }
            }
            return best;
        } else {
            int best = INT_MAX;
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
                if (g[i][j] == '.') {
                    g[i][j] = 'X';
                    best = min(best, minimax(g, depth + 1, true, alpha, beta));
                    g[i][j] = '.';
                    beta = min(beta, best);
                    if (beta <= alpha) return best;  // prune
                }
            }
            return best;
        }
    }

    pair<int,int> bestAIMove() {
        int best = INT_MIN; pair<int,int> move = {-1, -1};
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            if (b[i][j] == '.') {
                b[i][j] = 'O';
                int val = minimax(b, 0, false, INT_MIN, INT_MAX);
                b[i][j] = '.';
                if (val > best) { best = val; move = {i, j}; }
            }
        }
        return move;
    }

    void draw() {
        cout << "\n    0   1   2   3\n";
        for (int i = 0; i < N; i++) {
            cout << "  +---+---+---+---+\n";
            cout << i << " ";
            for (int j = 0; j < N; j++) {
                char c = b[i][j];
                cout << "| ";
                if (c == 'X') cout << C::GREEN << "X" << C::RESET;
                else if (c == 'O') cout << C::RED << "O" << C::RESET;
                else cout << " ";
                cout << " ";
            }
            cout << "|\n";
        }
        cout << "  +---+---+---+---+\n";
    }

    void play() {
        reset();
        cout << "\n" << C::BOLD << "== Room 4: Dark Lord's Chessboard ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: Minimax with Alpha-Beta Pruning\n"
             << "Data Structures: 2D board, implicit game tree\n"
             << "Time Complexity: O(b^d), ~O(b^(d/2)) with a/b pruning\n"
             << C::RESET;
        cout << "\nGet 4 in a row. You are " << C::GREEN << "X"
             << C::RESET << ", Dark Lord is " << C::RED << "O" << C::RESET
             << ". Enter moves as 'row col' (e.g. 1 2), or q to leave.\n";

        while (!over) {
            draw();
            cout << "Your move> ";
            string cmd; cin >> cmd;
            if (cmd == "q" || cmd == "Q") return;
            int r, c;
            try { r = stoi(cmd); cin >> c; } catch (...) { cout << "Bad input.\n"; continue; }
            if (r < 0 || r >= N || c < 0 || c >= N || b[r][c] != '.') {
                cout << C::RED << "Invalid move.\n" << C::RESET; continue;
            }
            b[r][c] = 'X';
            char w = winner(b);
            if (w == 'X') {
                draw();
                cout << C::GREEN << "\n** VICTORY! You defeated the Dark Lord! **\n"
                     << C::RESET;
                completeRoom(3, 250, 100, "Dark Lord's Crown");
                pauseMsg(); return;
            }
            if (isFull(b)) {
                draw();
                cout << "Draw! The Dark Lord retreats, wounded.\n";
                completeRoom(3, 150, 40);
                pauseMsg(); return;
            }
            cout << C::MAG << "The Dark Lord ponders...\n" << C::RESET;
            auto m = bestAIMove();
            if (m.first < 0) break;
            b[m.first][m.second] = 'O';
            cout << "Dark Lord plays " << m.first << " " << m.second << "\n";
            w = winner(b);
            if (w == 'O') {
                draw();
                hero.hp = max(0, hero.hp - 30);
                cout << C::RED << "\nDefeated... -30 HP\n" << C::RESET;
                completeRoom(3, 50, 0);
                pauseMsg(); return;
            }
            if (isFull(b)) {
                draw();
                cout << "Draw! The Dark Lord retreats, wounded.\n";
                completeRoom(3, 150, 40);
                pauseMsg(); return;
            }
        }
    }
};

// =====================================================================
// ROOM 5 - The Sorting Hall  (Quick Sort, Divide & Conquer)
// =====================================================================
struct SortingRoom {
    struct Scroll { int power; string name; };
    vector<Scroll> scrolls;
    long long comparisons = 0, swaps = 0;

    void init() {
        vector<string> names = {"Fire","Ice","Wind","Earth","Light","Shadow","Bolt","Water"};
        scrolls.clear();
        for (auto& n : names) scrolls.push_back({10 + rand() % 90, n});
        comparisons = swaps = 0;
    }

    bool isSorted() {
        for (size_t i = 1; i < scrolls.size(); i++)
            if (scrolls[i - 1].power > scrolls[i].power) return false;
        return true;
    }

    // Lomuto partition - pick rightmost as pivot.
    int partition(int lo, int hi) {
        int pivot = scrolls[hi].power;
        int i = lo - 1;
        for (int j = lo; j < hi; j++) {
            comparisons++;
            if (scrolls[j].power <= pivot) {
                i++;
                if (i != j) { swap(scrolls[i], scrolls[j]); swaps++; }
            }
        }
        swap(scrolls[i + 1], scrolls[hi]); swaps++;
        return i + 1;
    }

    // Quick Sort (Divide & Conquer). O(n log n) avg, O(n^2) worst.
    void quickSort(int lo, int hi) {
        if (lo < hi) {
            int p = partition(lo, hi);
            quickSort(lo, p - 1);
            quickSort(p + 1, hi);
        }
    }

    void draw() {
        cout << "\n  idx | name     | power\n";
        cout << "  ----+----------+------\n";
        for (size_t i = 0; i < scrolls.size(); i++)
            cout << "  " << setw(3) << i + 1 << " | "
                 << setw(8) << left << scrolls[i].name
                 << " | " << scrolls[i].power << "\n";
    }

    void play() {
        init();
        cout << "\n" << C::BOLD << "== Room 5: The Sorting Hall ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: Quick Sort (Divide & Conquer)\n"
             << "Data Structures: Array + recursion stack\n"
             << "Time Complexity: O(n log n) avg, O(n^2) worst\n"
             << C::RESET;
        cout << "\nSort scrolls ascending by power.\n"
             << "Commands: 'i j' swap idx i and j (1-based), c check (200pt),\n"
             << "          a auto-quicksort (80pt), q leave\n";

        while (true) {
            draw();
            cout << "> ";
            string cmd; cin >> cmd;
            if (cmd == "q" || cmd == "Q") return;
            if (cmd == "c" || cmd == "C") {
                if (isSorted()) {
                    cout << C::GREEN
                         << "** Perfectly sorted by hand! **\n"
                         << C::RESET;
                    completeRoom(4, 200, 60, "Master Sorter's Badge");
                    pauseMsg(); return;
                } else cout << C::RED << "Not sorted yet.\n" << C::RESET;
                continue;
            }
            if (cmd == "a" || cmd == "A") {
                quickSort(0, (int)scrolls.size() - 1);
                cout << "Sorted by QuickSort. Comparisons: " << comparisons
                     << "  Swaps: " << swaps << "\n";
                completeRoom(4, 80, 30);
                pauseMsg(); return;
            }
            try {
                int i = stoi(cmd);
                int j; cin >> j;
                if (i >= 1 && j >= 1 && i <= (int)scrolls.size() && j <= (int)scrolls.size()
                    && i != j) {
                    swap(scrolls[i - 1], scrolls[j - 1]);
                } else cout << "Bad indices.\n";
            } catch (...) { cout << "Unknown command.\n"; }
        }
    }
};

// =====================================================================
// ROOM 6 - The Arcane Lock  (N-Queens via Backtracking)
// =====================================================================
struct LockRoom {
    static const int N = 5;
    int board[N][N];
    int solution[N][N];
    int hintUsed = 0;

    bool isSafe(int b[N][N], int row, int col) {
        for (int i = 0; i < row; i++) if (b[i][col]) return false;
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--) if (b[i][j]) return false;
        for (int i = row, j = col; i >= 0 && j < N; i--, j++) if (b[i][j]) return false;
        return true;
    }

    // Classic N-Queens backtracking. Time O(N!), trimmed by pruning.
    bool solve(int b[N][N], int row) {
        if (row == N) return true;
        for (int col = 0; col < N; col++) {
            if (isSafe(b, row, col)) {
                b[row][col] = 1;
                if (solve(b, row + 1)) return true;
                b[row][col] = 0;  // BACKTRACK
            }
        }
        return false;
    }

    void init() {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) { board[i][j] = 0; solution[i][j] = 0; }
        solve(solution, 0);
        hintUsed = 0;
    }

    int countRunes() {
        int c = 0;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) if (board[i][j]) c++;
        return c;
    }

    bool isValid() {
        if (countRunes() != N) return false;
        vector<pair<int,int>> placed;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
            if (board[i][j]) placed.push_back({i, j});
        for (size_t a = 0; a < placed.size(); a++)
            for (size_t b = a + 1; b < placed.size(); b++) {
                auto [r1, c1] = placed[a]; auto [r2, c2] = placed[b];
                if (r1 == r2 || c1 == c2 || abs(r1 - r2) == abs(c1 - c2)) return false;
            }
        return true;
    }

    void draw() {
        cout << "\n    ";
        for (int j = 0; j < N; j++) cout << " " << j << " ";
        cout << "\n";
        for (int i = 0; i < N; i++) {
            cout << "  " << i << " ";
            for (int j = 0; j < N; j++)
                cout << (board[i][j] ? C::MAG + string(" * ") + C::RESET : " . ");
            cout << "\n";
        }
        cout << "Runes placed: " << countRunes() << "/" << N
             << "   Hints used: " << hintUsed << "\n";
    }

    void play() {
        init();
        cout << "\n" << C::BOLD << "== Room 6: The Arcane Lock ==" << C::RESET << "\n";
        cout << C::DIM << "Algorithm: N-Queens via Backtracking\n"
             << "Data Structures: 2D grid + recursion stack\n"
             << "Time Complexity: O(N!) worst case (pruning trims heavily)\n"
             << C::RESET;
        cout << "\nPlace " << N << " runes, no two on the same row / col / diagonal.\n"
             << "Commands: 'r c' toggle rune at row r, col c, "
             << "c check lock, h hint (-20), q leave\n";

        while (true) {
            draw();
            cout << "> ";
            string cmd; cin >> cmd;
            if (cmd == "q" || cmd == "Q") return;
            if (cmd == "c" || cmd == "C") {
                if (isValid()) {
                    int score = max(100, 300 - hintUsed * 20);
                    cout << C::GREEN << "\n** The lock opens! **\n" << C::RESET;
                    completeRoom(5, score, 100, "Master Key");
                    pauseMsg(); return;
                } else {
                    hero.hp = max(0, hero.hp - 5);
                    cout << C::RED << "Runes clash! -5 HP\n" << C::RESET;
                }
                continue;
            }
            if (cmd == "h" || cmd == "H") {
                hintUsed++;
                hero.score = max(0, hero.score - 20);
                for (int i = 0; i < N && true; i++)
                    for (int j = 0; j < N; j++)
                        if (solution[i][j] && !board[i][j]) {
                            cout << C::CYAN << "Try row " << i << ", col " << j
                                 << C::RESET << "\n";
                            goto hintDone;
                        }
                hintDone: continue;
            }
            try {
                int r = stoi(cmd); int c; cin >> c;
                if (r >= 0 && r < N && c >= 0 && c < N)
                    board[r][c] = board[r][c] ? 0 : 1;
                else cout << "Out of range.\n";
            } catch (...) { cout << "Unknown command.\n"; }
        }
    }
};

// =====================================================================
// Menu + leaderboard (priority_queue-backed Hall of Heroes)
// =====================================================================
void showLeaderboard() {
    priority_queue<LBEntry> pq;
    for (auto& e : leaderboard) pq.push(e);
    cout << "\n" << C::BOLD << "== Hall of Heroes ==" << C::RESET
         << C::DIM << "   (Max-Heap, O(log n) per insert)\n" << C::RESET;
    int rank = 1;
    while (!pq.empty() && rank <= 10) {
        auto e = pq.top(); pq.pop();
        const char* medal = rank == 1 ? "[1]" : rank == 2 ? "[2]" : rank == 3 ? "[3]" : "   ";
        cout << "  " << medal << " " << setw(16) << left << e.name
             << "  score " << setw(5) << e.score
             << "  rooms " << e.rooms << "/6\n";
        rank++;
    }
    pauseMsg();
}

void rest() {
    if (hero.gold < 20) { cout << C::RED << "Not enough gold!\n" << C::RESET; return; }
    hero.gold -= 20;
    hero.hp = min(hero.maxHp, hero.hp + 50);
    cout << C::GREEN << "You rest. +50 HP\n" << C::RESET;
}

int main() {
    srand((unsigned)time(nullptr));

    // Intro
    cout << C::CYAN
         << "\n  === AlgoQuest: The Algorithmic Dungeon ===\n"
         << "  6 rooms, 6 classic algorithms, multiple paradigms.\n"
         << C::RESET;
    cout << "\nEnter hero name: ";
    string name; getline(cin, name);
    if (!name.empty()) hero.name = name;

    MazeRoom     r1;
    TreasureRoom r2;
    BridgeRoom   r3;
    ChessRoom    r4;
    SortingRoom  r5;
    LockRoom     r6;

    vector<string> roomNames = {
        "1. The Haunted Maze        (BFS / DFS       - Graph Traversal)",
        "2. The Treasure Vault      (0/1 Knapsack    - Dynamic Programming)",
        "3. Bridge of Merchants     (Kruskal's MST   - Greedy + DSU)",
        "4. Dark Lord's Chessboard  (Minimax + a-b   - Game Theory)",
        "5. The Sorting Hall        (Quick Sort      - Divide & Conquer)",
        "6. The Arcane Lock         (N-Queens        - Backtracking)",
    };

    while (true) {
        showStats();
        cout << "\nChoose a trial:\n";
        for (size_t i = 0; i < roomNames.size(); i++) {
            cout << "  " << (hero.cleared[i] ? C::GREEN + string("[x] ") + C::RESET : "[ ] ")
                 << roomNames[i] << "\n";
        }
        cout << "  L. Leaderboard\n  R. Rest (20 gold -> +50 HP)\n";
        if (hero.rooms == 6) cout << "  V. " << C::GREEN << "Claim Victory!" << C::RESET << "\n";
        cout << "  Q. Quit\n> ";

        string cmd; cin >> cmd;
        if (cmd.empty()) continue;
        char c0 = toupper(cmd[0]);

        if (c0 == 'Q') {
            leaderboard.push_back({hero.name, hero.score, hero.rooms});
            cout << "\nFinal score: " << hero.score << "\n";
            break;
        }
        if (c0 == 'L') { showLeaderboard(); continue; }
        if (c0 == 'R') { rest(); continue; }
        if (c0 == 'V' && hero.rooms == 6) {
            int final_ = hero.score + 500;
            leaderboard.push_back({hero.name, final_, hero.rooms});
            cout << C::GREEN << "\n*** VICTORY *** Final score (incl. +500 bonus): "
                 << final_ << "\n" << C::RESET;
            break;
        }
        if (c0 >= '1' && c0 <= '6') {
            int idx = c0 - '1';
            if (hero.cleared[idx]) {
                cout << C::DIM << "Already cleared.\n" << C::RESET;
                continue;
            }
            switch (idx) {
                case 0: r1.play(); break;
                case 1: r2.play(); break;
                case 2: r3.play(); break;
                case 3: r4.play(); break;
                case 4: r5.play(); break;
                case 5: r6.play(); break;
            }
            if (hero.hp <= 0) {
                cout << C::RED << "\nYou have fallen...\n" << C::RESET;
                leaderboard.push_back({hero.name, hero.score, hero.rooms});
                break;
            }
        } else {
            cout << "Unknown command.\n";
        }
    }

    showLeaderboard();
    return 0;
}
