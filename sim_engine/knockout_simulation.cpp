// Compile: g++ -std=c++20 simulate_knockout.cpp -o simulate_knockout
// Run: ./simulate_knockout

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <array>

using namespace std;

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------
struct MatchInfo {
    string home, away;
    double home_xg, away_xg;
    bool is_played;
};

struct ProbInfo {
    double p_home_win, p_draw, p_away_win;
};

struct MatchResult {
    int id;
    string phase;
    string team1, team2;
    double home_xg, away_xg;
    bool is_played;
    double t1_win_prob, draw_prob, t2_win_prob;
    bool extra_time;
    bool penalties;
    string final_score;                 // score after 90 or ET (before penalties)
    string winner;
    string loser;
};

// Global storage for JSON output
vector<MatchResult> allMatches;

// ---------------------------------------------------------
// MATHS HELPERS
// ---------------------------------------------------------
double factorial(int n) {
    double res = 1.0;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

double poisson_prob(int k, double lambda) {
    if (lambda <= 0.0) lambda = 0.1;
    return pow(lambda, k) * exp(-lambda) / factorial(k);
}

// Most likely score unconditionally
pair<int,int> most_likely_score(double home_xg, double away_xg, int max_goals = 10) {
    pair<int,int> best{0,0};
    double best_prob = 0.0;
    for (int h = 0; h <= max_goals; ++h) {
        for (int a = 0; a <= max_goals; ++a) {
            double prob = poisson_prob(h, home_xg) * poisson_prob(a, away_xg);
            if (prob > best_prob) {
                best_prob = prob;
                best = {h, a};
            }
        }
    }
    return best;
}

// Most likely score satisfying a given condition (H = home win, A = away win, D = draw)
pair<int,int> most_likely_conditional(double hx, double ax, char condition, int max_g = 10) {
    pair<int,int> best = {0,0};
    double best_p = -1.0;
    for (int h = 0; h <= max_g; h++) {
        for (int a = 0; a <= max_g; a++) {
            double prob = poisson_prob(h, hx) * poisson_prob(a, ax);
            bool match = false;
            if (condition == 'H' && h > a) match = true;
            if (condition == 'A' && a > h) match = true;
            if (condition == 'D' && h == a) match = true;
            if (match && prob > best_p) {
                best_p = prob;
                best = {h, a};
            }
        }
    }
    return best;
}

// ---------------------------------------------------------
// PENALTY SHOOTOUT
// ---------------------------------------------------------
mt19937 rng(random_device{}());

bool penalty_shootout_winner(const string& t1, const string& t2,
                             double home_xg, double away_xg,
                             bool t1_is_home) {
    double diff = home_xg - away_xg;
    double probA = 0.75 + clamp(diff * 0.05, -0.1, 0.1);
    double probB = 0.75 - clamp(diff * 0.05, -0.1, 0.1);
    if (!t1_is_home) swap(probA, probB);
    probA = clamp(probA, 0.6, 0.9);
    probB = clamp(probB, 0.6, 0.9);

    uniform_real_distribution<double> dist(0.0, 1.0);
    auto shoot = [&](double p) { return dist(rng) < p; };

    int scoreA = 0, scoreB = 0;
    for (int round = 0; round < 5; ++round) {
        if (shoot(probA)) scoreA++;
        if (shoot(probB)) scoreB++;
        int remainA = 5 - round - 1;
        int remainB = (round < 4) ? 5 - round - 1 : 0;
        if (scoreA > scoreB + remainB) return true;
        if (scoreB > scoreA + remainA) return false;
    }
    while (true) {
        bool a = shoot(probA), b = shoot(probB);
        if (a && !b) return true;
        if (!a && b) return false;
    }
}

// ---------------------------------------------------------
// CSV LOADERS
// ---------------------------------------------------------
unordered_map<string, MatchInfo> xg_table;
unordered_map<string, ProbInfo> prob_table;

void load_xg_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening " << filename << endl;
        exit(1);
    }
    string line;
    getline(file, line); // skip header
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string home, away, hxg_str, axg_str, played_str;
        getline(ss, home, ',');
        getline(ss, away, ',');
        getline(ss, hxg_str, ',');
        getline(ss, axg_str, ',');
        getline(ss, played_str, ',');

        double hx = stod(hxg_str);
        double ax = stod(axg_str);
        bool played = (stoi(played_str) == 1);

        string key = home + "|" + away;
        xg_table[key] = {home, away, hx, ax, played};
        key = away + "|" + home;
        xg_table[key] = {away, home, ax, hx, played};
    }
    cout << "Loaded " << xg_table.size() / 2 << " unique matchups from xG table.\n";
}

void load_prob_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening " << filename << endl;
        exit(1);
    }
    string line;
    getline(file, line); // skip header
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string home, away, hxg, axg, phw, pd, paw, played;
        getline(ss, home, ',');
        getline(ss, away, ',');
        getline(ss, hxg, ',');   // Home_xG (ignore, already in xg_table)
        getline(ss, axg, ',');   // Away_xG (ignore)
        getline(ss, phw, ',');
        getline(ss, pd, ',');
        getline(ss, paw, ',');
        // last field is Is_Played, ignore

        double p_home = stod(phw), p_draw = stod(pd), p_away = stod(paw);
        string key = home + "|" + away;
        prob_table[key] = {p_home, p_draw, p_away};
        key = away + "|" + home;
        prob_table[key] = {p_away, p_draw, p_home};
    }
    cout << "Loaded probabilities for " << prob_table.size() / 2 << " matchups.\n";
}

MatchInfo get_match(const string& t1, const string& t2) {
    string key = t1 + "|" + t2;
    if (xg_table.count(key)) return xg_table[key];
    key = t2 + "|" + t1;
    if (xg_table.count(key)) return xg_table[key];
    cerr << "Error: no xG data for " << t1 << " vs " << t2 << endl;
    exit(1);
}

ProbInfo get_prob(const string& t1, const string& t2) {
    string key = t1 + "|" + t2;
    if (prob_table.count(key)) return prob_table[key];
    key = t2 + "|" + t1;
    if (prob_table.count(key)) return prob_table[key];
    cerr << "Error: no probability data for " << t1 << " vs " << t2 << endl;
    exit(1);
}

// ---------------------------------------------------------
// MATCH SIMULATOR (uses calibrated probabilities + realistic score)
// ---------------------------------------------------------
// Replace the old simulate_match with this version

pair<string,string> simulate_match(int matchId, const string& phase,
                                   const string& t1, const string& t2) {
    MatchInfo info = get_match(t1, t2);
    ProbInfo prob = get_prob(t1, t2);

    MatchResult result;
    result.id = matchId;
    result.phase = phase;
    result.team1 = t1;
    result.team2 = t2;
    result.home_xg = info.home_xg;
    result.away_xg = info.away_xg;
    result.is_played = info.is_played;
    result.extra_time = false;
    result.penalties = false;

    result.t1_win_prob = prob.p_home_win;
    result.draw_prob   = prob.p_draw;
    result.t2_win_prob = prob.p_away_win;

    int goals1, goals2;
    string suffix;
    string winner;

    // Helper: unconditional most likely score for a given xG pair
    auto most_likely_score_fn = [](double hx, double ax, int max_g = 10) -> pair<int,int> {
        pair<int,int> best{0,0};
        double best_p = -1.0;
        for (int h = 0; h <= max_g; ++h) {
            for (int a = 0; a <= max_g; ++a) {
                double p = poisson_prob(h, hx) * poisson_prob(a, ax);
                if (p > best_p) {
                    best_p = p;
                    best = {h, a};
                }
            }
        }
        return best;
    };

    if (info.is_played) {
        // Already played – use rounded xG
        goals1 = round(info.home_xg);
        goals2 = round(info.away_xg);
        suffix = "(played)";
        if (goals1 > goals2) winner = info.home;
        else if (goals2 > goals1) winner = info.away;
        else {
            result.penalties = true;
            suffix = "(played, tied)";
            winner = (prob.p_home_win >= prob.p_away_win) ? info.home : info.away;
        }
    } else {
        // --- 90 minutes: unconditional most likely score ---
        auto [h90, a90] = most_likely_score_fn(info.home_xg, info.away_xg);
        goals1 = h90;
        goals2 = a90;

        if (goals1 != goals2) {
            suffix = "90'";
            winner = (goals1 > goals2) ? info.home : info.away;
        } else {
            // --- Extra time ---
            result.extra_time = true;
            double etxg_h = info.home_xg * 0.5;
            double etxg_a = info.away_xg * 0.5;
            auto [h_et, a_et] = most_likely_score_fn(etxg_h, etxg_a, 8); // max 8 for ET
            goals1 = h90 + h_et;
            goals2 = a90 + a_et;

            if (goals1 != goals2) {
                suffix = "AET";
                winner = (goals1 > goals2) ? info.home : info.away;
            } else {
                // --- Penalties ---
                result.penalties = true;
                suffix = "pens";
                winner = (prob.p_home_win >= prob.p_away_win) ? info.home : info.away;
            }
        }

        // Print the probability line as before
        cout << "       ["
             << t1 << " " << fixed << setprecision(0) << prob.p_home_win*100 << "%  "
             << "Draw " << prob.p_draw*100 << "%  "
             << t2 << " " << prob.p_away_win*100 << "%]\n";
    }

    result.final_score = to_string(goals1) + "-" + to_string(goals2);
    result.winner = winner;
    result.loser = (winner == t1) ? t2 : t1;

    allMatches.push_back(result);

    cout << left << setw(9) << ("[" + phase + "]")
         << "M" << setw(4) << matchId << ": "
         << setw(22) << t1 << " vs " << setw(22) << t2 << " → "
         << "Prediction: " << result.final_score
         << " (" << suffix << ")";

    if (result.penalties)
        cout << " [Penalties → higher win prob: " << winner << " advances]";
    else
        cout << " [" << winner << " advances]";

    cout << "\n";

    return {winner, result.loser};
}
// ---------------------------------------------------------
// JSON OUTPUT BUILDER
// ---------------------------------------------------------
void write_json(const string& filename, const string& champion,
                const string& runner_up, const string& third_place,
                const string& fourth_place) {
    ofstream out(filename);
    if (!out.is_open()) {
        cerr << "Error writing JSON to " << filename << endl;
        return;
    }
    out << "{\n";
    out << "  \"matches\": [\n";
    for (size_t i = 0; i < allMatches.size(); ++i) {
        const auto& m = allMatches[i];
        out << "    {\n";
        out << "      \"id\": " << m.id << ",\n";
        out << "      \"phase\": \"" << m.phase << "\",\n";
        out << "      \"team1\": \"" << m.team1 << "\",\n";
        out << "      \"team2\": \"" << m.team2 << "\",\n";
        out << "      \"home_xg\": " << m.home_xg << ",\n";
        out << "      \"away_xg\": " << m.away_xg << ",\n";
        out << "      \"is_played\": " << (m.is_played ? "true" : "false") << ",\n";
        out << "      \"90min_probabilities\": {\n";
        out << "        \"team1_win\": " << fixed << setprecision(4) << m.t1_win_prob << ",\n";
        out << "        \"draw\": " << m.draw_prob << ",\n";
        out << "        \"team2_win\": " << m.t2_win_prob << "\n";
        out << "      },\n";
        out << "      \"final_score\": \"" << m.final_score << "\",\n";
        out << "      \"extra_time\": " << (m.extra_time ? "true" : "false") << ",\n";
        out << "      \"penalties\": " << (m.penalties ? "true" : "false") << ",\n";
        out << "      \"winner\": \"" << m.winner << "\",\n";
        out << "      \"loser\": \"" << m.loser << "\"\n";
        out << "    }" << (i < allMatches.size() - 1 ? "," : "") << "\n";
    }
    out << "  ],\n";
    out << "  \"podium\": {\n";
    out << "    \"winner\": \"" << champion << "\",\n";
    out << "    \"runner_up\": \"" << runner_up << "\",\n";
    out << "    \"third_place\": \"" << third_place << "\",\n";
    out << "    \"fourth_place\": \"" << fourth_place << "\"\n";
    out << "  }\n";
    out << "}\n";
    out.close();
    cout << "\n✅ JSON results saved to " << filename << "\n";
}

// ---------------------------------------------------------
// MAIN – official 2026 bracket
// ---------------------------------------------------------
int main() {
    cout << "================================================================================\n";
    cout << "  FIFA WORLD CUP 2026 – KNOCKOUT STAGE PREDICTOR (Calibrated Probabilities)\n";
    cout << "  Uses calibrated win/draw/loss probabilities + most‑likely conditional score\n";
    cout << "  Extra time xG = 50% of 90‑minute xG\n";
    cout << "================================================================================\n";

    // Load both CSV files
    load_xg_csv("../data/knockout_xg_lookup.csv");
    load_prob_csv("../data/knockout_prob_lookup.csv");

    cout << "\n--- ROUND OF 32 ---\n";
    auto m73  = simulate_match(73,  "R32", "South Africa", "Canada");
    auto m74  = simulate_match(74,  "R32", "Germany", "Paraguay");
    auto m75  = simulate_match(75,  "R32", "Netherlands", "Morocco");
    auto m76  = simulate_match(76,  "R32", "Brazil", "Japan");
    auto m77  = simulate_match(77,  "R32", "France", "Sweden");
    auto m78  = simulate_match(78,  "R32", "Ivory Coast", "Norway");
    auto m79  = simulate_match(79,  "R32", "Mexico", "Ecuador");
    auto m80  = simulate_match(80,  "R32", "England", "DR Congo");
    auto m81  = simulate_match(81,  "R32", "United States", "Bosnia and Herzegovina");
    auto m82  = simulate_match(82,  "R32", "Belgium", "Senegal");
    auto m83  = simulate_match(83,  "R32", "Portugal", "Croatia");
    auto m84  = simulate_match(84,  "R32", "Spain", "Austria");
    auto m85  = simulate_match(85,  "R32", "Switzerland", "Algeria");
    auto m86  = simulate_match(86,  "R32", "Argentina", "Cape Verde");
    auto m87  = simulate_match(87,  "R32", "Colombia", "Ghana");
    auto m88  = simulate_match(88,  "R32", "Australia", "Egypt");

    cout << "\n--- ROUND OF 16 ---\n";
    auto m89  = simulate_match(89,  "R16", m74.first, m77.first);
    auto m90  = simulate_match(90,  "R16", m73.first, m75.first);
    auto m91  = simulate_match(91,  "R16", m76.first, m78.first);
    auto m92  = simulate_match(92,  "R16", m79.first, m80.first);
    auto m93  = simulate_match(93,  "R16", m83.first, m84.first);
    auto m94  = simulate_match(94,  "R16", m81.first, m82.first);
    auto m95  = simulate_match(95,  "R16", m86.first, m88.first);
    auto m96  = simulate_match(96,  "R16", m85.first, m87.first);

    cout << "\n--- QUARTER‑FINALS ---\n";
    auto m97  = simulate_match(97,  "QF", m89.first, m90.first);
    auto m98  = simulate_match(98,  "QF", m93.first, m94.first);
    auto m99  = simulate_match(99,  "QF", m91.first, m92.first);
    auto m100 = simulate_match(100, "QF", m95.first, m96.first);

    cout << "\n--- SEMI‑FINALS ---\n";
    auto m101 = simulate_match(101, "SF", m97.first, m98.first);
    auto m102 = simulate_match(102, "SF", m99.first, m100.first);

    cout << "\n--- THIRD PLACE MATCH ---\n";
    auto m103 = simulate_match(103, "3rd", m101.second, m102.second);

    cout << "\n--- FINAL ---\n";
    auto m104 = simulate_match(104, "Final", m101.first, m102.first);

    string champion = m104.first;
    string runner_up = (champion == m101.first) ? m102.first : m101.first;
    string third_place = m103.first;
    string fourth_place = m103.second;

    cout << "\n================================================================================\n";
    cout << "  WINNER:       " << champion << "\n";
    cout << "  RUNNER-UP:    " << runner_up << "\n";
    cout << "  THIRD PLACE:  " << third_place << "\n";
    cout << "================================================================================\n";

    // Write JSON results
    write_json("../data/knockout_simulation_results.json", champion, runner_up, third_place, fourth_place);

    return 0;
}