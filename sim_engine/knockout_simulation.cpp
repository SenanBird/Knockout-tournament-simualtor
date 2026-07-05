// Compile: g++ -std=c++20 -fopenmp simulate_knockout.cpp -o simulate_knockout
// Run: ./simulate_knockout

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <array>
#include <omp.h>

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

struct KnockoutMatch {
    int home_goals, away_goals;
    string date;                // YYYY-MM-DD
};

enum SlotType { TEAM, WINNER };
struct Slot {
    SlotType type;
    string team_name;           // TEAM
    int match_id;               // WINNER
};

struct BracketMatch {
    int id;
    string phase;
    Slot team1;
    Slot team2;
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
    string final_score;
    string winner;
    string loser;
};

// ---------------------------------------------------------
// GLOBAL STATE (read‑only after loading)
// ---------------------------------------------------------
unordered_map<string, pair<double,double>> xg_table;
unordered_map<string, ProbInfo> prob_table;
unordered_map<string, int> alive_status;
unordered_map<string, KnockoutMatch> actual_scores;

// ---------------------------------------------------------
// MATH HELPERS
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

// ---------------------------------------------------------
// PENALTY SHOOTOUT (thread‑safe: uses a local RNG)
// ---------------------------------------------------------
bool penalty_shootout_winner_local(const string& t1, const string& t2,
                                   double home_xg, double away_xg,
                                   bool t1_is_home, mt19937& rng) {
    double diff = home_xg - away_xg;
    double probA = clamp(0.75 + diff * 0.05, 0.65, 0.85);
    double probB = clamp(0.75 - diff * 0.05, 0.65, 0.85);
    if (!t1_is_home) swap(probA, probB);

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
// FILE LOADERS
// ---------------------------------------------------------
void load_prob_csv(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Error opening " << filename << endl; exit(1); }
    string line;
    getline(file, line);
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string home, away, hxg_str, axg_str, phw, pd, paw, played;
        getline(ss, home, ',');
        getline(ss, away, ',');
        getline(ss, hxg_str, ',');
        getline(ss, axg_str, ',');
        getline(ss, phw, ',');
        getline(ss, pd, ',');
        getline(ss, paw, ',');
        getline(ss, played, ',');

        double hx = stod(hxg_str);
        double ax = stod(axg_str);
        double p_home = stod(phw);
        double p_draw = stod(pd);
        double p_away = stod(paw);

        string key = home + "|" + away;
        xg_table[key] = {hx, ax};
        key = away + "|" + home;
        xg_table[key] = {ax, hx};

        key = home + "|" + away;
        prob_table[key] = {p_home, p_draw, p_away};
        key = away + "|" + home;
        prob_table[key] = {p_away, p_draw, p_home};
    }
    cout << "Loaded " << prob_table.size() / 2 << " matchups (xG + probabilities).\n";
}

void load_alive(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Warning: could not open " << filename << endl; return; }
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string team, status_str;
        getline(ss, team, ',');
        getline(ss, status_str, ',');
        alive_status[team] = stoi(status_str);
    }
    cout << "Loaded alive status for " << alive_status.size() << " teams.\n";
}

void load_actual_results(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) { cerr << "Warning: could not open " << filename << endl; return; }
    string line;
    getline(file, line);
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string date, home, away, hg_str, ag_str, tournament, rest;
        getline(ss, date, ',');
        getline(ss, home, ',');
        getline(ss, away, ',');
        getline(ss, hg_str, ',');
        getline(ss, ag_str, ',');
        getline(ss, tournament, ',');
        if (tournament != "FIFA World Cup") continue;
        if (date < "2026-06-28") continue;
        if (hg_str == "NA" || ag_str == "NA") continue;

        int hg = stoi(hg_str);
        int ag = stoi(ag_str);
        string key = home + "|" + away;
        actual_scores[key] = {hg, ag, date};
        key = away + "|" + home;
        actual_scores[key] = {ag, hg, date};
    }
    cout << "Loaded " << actual_scores.size() / 2 << " knockout results.\n";
}

// ---------------------------------------------------------
// HELPERS (thread‑safe – only read global data)
// ---------------------------------------------------------
bool is_eliminated(const string& team) {
    return alive_status.count(team) && alive_status[team] == 1;
}

bool get_actual_result(const string& t1, const string& t2,
                       int& goals1, int& goals2, string& winner) {
    string key = t1 + "|" + t2;
    auto it = actual_scores.find(key);
    if (it == actual_scores.end()) return false;

    goals1 = it->second.home_goals;
    goals2 = it->second.away_goals;

    if (goals1 > goals2) { winner = t1; return true; }
    if (goals2 > goals1) { winner = t2; return true; }

    string draw_date = it->second.date;
    bool t1_later = false, t2_later = false;
    for (const auto& [k, m] : actual_scores) {
        if (m.date > draw_date) {
            size_t sep = k.find('|');
            string h = k.substr(0, sep);
            string a = k.substr(sep + 1);
            if (h == t1 || a == t1) t1_later = true;
            if (h == t2 || a == t2) t2_later = true;
        }
    }
    if (t1_later && !t2_later) { winner = t1; return true; }
    if (!t1_later && t2_later) { winner = t2; return true; }

    bool t1_out = is_eliminated(t1);
    bool t2_out = is_eliminated(t2);
    if (!t1_out && t2_out) { winner = t1; return true; }
    if (t1_out && !t2_out) { winner = t2; return true; }

    cerr << "Error: draw between " << t1 << " and " << t2
         << " and no definitive advancement data.\n";
    return false;
}

MatchInfo get_match(const string& t1, const string& t2) {
    string key = t1 + "|" + t2;
    if (xg_table.count(key)) {
        auto [hx, ax] = xg_table[key];
        return {t1, t2, hx, ax, false};
    }
    key = t2 + "|" + t1;
    if (xg_table.count(key)) {
        auto [hx, ax] = xg_table[key];
        return {t2, t1, ax, hx, false};
    }
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
// RANDOM MATCH – thread‑safe version (uses a local RNG)
// ---------------------------------------------------------
string random_match_winner_local(const string& t1, const string& t2,
                                 int& goals1, int& goals2, bool& pens,
                                 mt19937& rng) {
    int g1, g2; string w;
    if (get_actual_result(t1, t2, g1, g2, w)) {
        goals1 = g1; goals2 = g2;
        pens = (g1 == g2);
        return w;
    }
    if (is_eliminated(t1) && !is_eliminated(t2)) {
        goals1 = 0; goals2 = 3; pens = false; return t2;
    }
    if (!is_eliminated(t1) && is_eliminated(t2)) {
        goals1 = 3; goals2 = 0; pens = false; return t1;
    }

    MatchInfo info = get_match(t1, t2);
    poisson_distribution<int> home_goals(info.home_xg);
    poisson_distribution<int> away_goals(info.away_xg);
    goals1 = home_goals(rng);
    goals2 = away_goals(rng);
    pens = false;
    if (goals1 != goals2)
        return (goals1 > goals2) ? info.home : info.away;

    // extra time
    poisson_distribution<int> et_home(info.home_xg * 0.5);
    poisson_distribution<int> et_away(info.away_xg * 0.5);
    goals1 += et_home(rng);
    goals2 += et_away(rng);
    if (goals1 != goals2)
        return (goals1 > goals2) ? info.home : info.away;

    // penalties
    pens = true;
    return penalty_shootout_winner_local(info.home, info.away,
                                         info.home_xg, info.away_xg, true, rng)
               ? info.home : info.away;
}

// ---------------------------------------------------------
// DETERMINISTIC MATCH (unchanged, uses original global‑dependent
// get_actual_result which is safe)
// ---------------------------------------------------------
void deterministic_display_match(int matchId, const string& phase,
                                 const string& t1, const string& t2,
                                 vector<MatchResult>& results) {
    MatchResult res;
    res.id = matchId;
    res.phase = phase;
    res.team1 = t1;
    res.team2 = t2;
    res.extra_time = false;
    res.penalties = false;
    res.t1_win_prob = res.draw_prob = res.t2_win_prob = 0.0;

    int goals1, goals2;
    string winner, suffix;

    if (get_actual_result(t1, t2, goals1, goals2, winner)) {
        suffix = "(played)";
        res.is_played = true;
        res.home_xg = goals1;
        res.away_xg = goals2;
        if (goals1 == goals2) res.penalties = true;
    } else {
        MatchInfo info = get_match(t1, t2);
        ProbInfo prob = get_prob(t1, t2);
        res.home_xg = info.home_xg;
        res.away_xg = info.away_xg;
        res.is_played = false;
        res.t1_win_prob = prob.p_home_win;
        res.draw_prob   = prob.p_draw;
        res.t2_win_prob = prob.p_away_win;

        auto [h90, a90] = most_likely_score(info.home_xg, info.away_xg);
        goals1 = h90; goals2 = a90;
        if (goals1 != goals2) {
            suffix = "90'";
            winner = (goals1 > goals2) ? info.home : info.away;
        } else {
            res.extra_time = true;
            double etxg_h = info.home_xg * 0.5;
            double etxg_a = info.away_xg * 0.5;
            auto [h_et, a_et] = most_likely_score(etxg_h, etxg_a, 8);
            goals1 = h90 + h_et; goals2 = a90 + a_et;
            if (goals1 != goals2) {
                suffix = "AET";
                winner = (goals1 > goals2) ? info.home : info.away;
            } else {
                res.penalties = true;
                suffix = "pens";
                winner = (prob.p_home_win >= prob.p_away_win) ? info.home : info.away;
            }
        }
    }

    res.final_score = to_string(goals1) + "-" + to_string(goals2);
    res.winner = winner;
    res.loser = (winner == t1) ? t2 : t1;
    results.push_back(res);
}

// ---------------------------------------------------------
// BRACKET RESOLVER
// ---------------------------------------------------------
string resolve_slot(const Slot& slot, const unordered_map<int,string>& winners) {
    if (slot.type == TEAM)
        return slot.team_name;
    else
        return winners.at(slot.match_id);
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main() {
    cout << "================================================================================\n";
    cout << "  FIFA WORLD CUP 2026 – KNOCKOUT PREDICTOR (Consensus Bracket)\n";
    cout << "================================================================================\n";

    load_prob_csv("../data/knockout_prob_lookup.csv");
    load_alive("../data/alive_teams.txt");
    load_actual_results("../Training Data/results.csv");

    // Official bracket (2026 knockout pairings)
    vector<BracketMatch> bracket = {
        // R32
        {73, "R32", {TEAM, "South Africa", 0},          {TEAM, "Canada", 0}},
        {74, "R32", {TEAM, "Germany", 0},               {TEAM, "Paraguay", 0}},
        {75, "R32", {TEAM, "Netherlands", 0},           {TEAM, "Morocco", 0}},
        {76, "R32", {TEAM, "Brazil", 0},                {TEAM, "Japan", 0}},
        {77, "R32", {TEAM, "France", 0},                {TEAM, "Sweden", 0}},
        {78, "R32", {TEAM, "Ivory Coast", 0},           {TEAM, "Norway", 0}},
        {79, "R32", {TEAM, "Mexico", 0},                {TEAM, "Ecuador", 0}},
        {80, "R32", {TEAM, "England", 0},               {TEAM, "DR Congo", 0}},
        {81, "R32", {TEAM, "United States", 0},         {TEAM, "Bosnia and Herzegovina", 0}},
        {82, "R32", {TEAM, "Belgium", 0},               {TEAM, "Senegal", 0}},
        {83, "R32", {TEAM, "Portugal", 0},              {TEAM, "Croatia", 0}},
        {84, "R32", {TEAM, "Spain", 0},                 {TEAM, "Austria", 0}},
        {85, "R32", {TEAM, "Switzerland", 0},           {TEAM, "Algeria", 0}},
        {86, "R32", {TEAM, "Argentina", 0},             {TEAM, "Cape Verde", 0}},
        {87, "R32", {TEAM, "Colombia", 0},              {TEAM, "Ghana", 0}},
        {88, "R32", {TEAM, "Australia", 0},             {TEAM, "Egypt", 0}},
        // R16
        {89, "R16", {WINNER, "", 74},  {WINNER, "", 77}},
        {90, "R16", {WINNER, "", 73},  {WINNER, "", 75}},
        {91, "R16", {WINNER, "", 76},  {WINNER, "", 78}},
        {92, "R16", {WINNER, "", 79},  {WINNER, "", 80}},
        {93, "R16", {WINNER, "", 83},  {WINNER, "", 84}},
        {94, "R16", {WINNER, "", 81},  {WINNER, "", 82}},
        {95, "R16", {WINNER, "", 86},  {WINNER, "", 88}},
        {96, "R16", {WINNER, "", 85},  {WINNER, "", 87}},
        // QF
        {97,  "QF", {WINNER, "", 89},  {WINNER, "", 90}},
        {98,  "QF", {WINNER, "", 93},  {WINNER, "", 94}},
        {99,  "QF", {WINNER, "", 91},  {WINNER, "", 92}},
        {100, "QF", {WINNER, "", 95},  {WINNER, "", 96}},
        // SF
        {101, "SF", {WINNER, "", 97},  {WINNER, "", 98}},
        {102, "SF", {WINNER, "", 99},  {WINNER, "", 100}},
        // 3rd place & Final
        {103, "3rd",   {WINNER, "", 101}, {WINNER, "", 102}},
        {104, "Final", {WINNER, "", 101}, {WINNER, "", 102}}
    };

    // ---------------------------------------------------------
    // 1. MONTE CARLO SIMULATION (OpenMP parallel)
    // ---------------------------------------------------------
    const int SIMS = 1000000;
    cout << "\nRunning " << SIMS << " Monte Carlo simulations";
    cout << " on " << omp_get_max_threads() << " threads...\n";

    // Global accumulators (will be filled after parallel section)
    unordered_map<string, array<int,5>> team_stats;
    unordered_map<int, unordered_map<string,int>> winner_counts;

    #pragma omp parallel
    {
        // Each thread has its own RNG and local counters
        mt19937 local_rng(random_device{}() + omp_get_thread_num());
        unordered_map<string, array<int,5>> local_team_stats;
        unordered_map<int, unordered_map<string,int>> local_winner_counts;

        // Initialize local team stats with all R32 teams
        for (const auto& m : bracket)
            if (m.phase == "R32") {
                local_team_stats[m.team1.team_name] = {0};
                local_team_stats[m.team2.team_name] = {0};
            }

        #pragma omp for schedule(static)
        for (int sim = 0; sim < SIMS; ++sim) {
            unordered_map<int,string> mc_winners;
            unordered_map<int,string> mc_losers;

            for (const auto& m : bracket) {
                if (m.phase == "3rd") continue;
                string t1 = resolve_slot(m.team1, mc_winners);
                string t2 = resolve_slot(m.team2, mc_winners);
                int g1, g2; bool pens;
                string winner = random_match_winner_local(t1, t2, g1, g2, pens, local_rng);
                mc_winners[m.id] = winner;
                mc_losers[m.id] = (winner == t1) ? t2 : t1;

                if (m.phase == "R32")      local_team_stats[winner][0]++;
                else if (m.phase == "R16") local_team_stats[winner][1]++;
                else if (m.phase == "QF")  local_team_stats[winner][2]++;
                else if (m.phase == "SF")  local_team_stats[winner][3]++;
            }

            // 3rd place and final
            int sf1 = 101, sf2 = 102;
            string t1 = mc_losers[sf1];
            string t2 = mc_losers[sf2];
            int g1, g2; bool pens;
            string third = random_match_winner_local(t1, t2, g1, g2, pens, local_rng);
            mc_winners[103] = third;

            string finalist1 = mc_winners[101];
            string finalist2 = mc_winners[102];
            string champion = random_match_winner_local(finalist1, finalist2, g1, g2, pens, local_rng);
            mc_winners[104] = champion;
            local_team_stats[champion][4]++;

            for (const auto& m : bracket) {
                local_winner_counts[m.id][mc_winners[m.id]]++;
            }
        }

        // Merge thread‑local results into global structures
        #pragma omp critical
        {
            for (const auto& [team, stats] : local_team_stats)
                for (int i = 0; i < 5; ++i)
                    team_stats[team][i] += stats[i];

            for (const auto& [match_id, counts] : local_winner_counts)
                for (const auto& [team, count] : counts)
                    winner_counts[match_id][team] += count;
        }
    }

    cout << "Simulation complete.\n";

    // ---------------------------------------------------------
    // 2. BUILD CONSENSUS BRACKET (sequential)
    // ---------------------------------------------------------
    unordered_map<int,string> consensus_winners;
    for (const auto& m : bracket) {
        if (m.phase == "3rd") continue;
        string t1 = resolve_slot(m.team1, consensus_winners);
        string t2 = resolve_slot(m.team2, consensus_winners);
        int g1, g2; string played_winner;
        if (get_actual_result(t1, t2, g1, g2, played_winner)) {
            consensus_winners[m.id] = played_winner;
        } else {
            auto& counts = winner_counts[m.id];
            int cnt1 = counts[t1];
            int cnt2 = counts[t2];
            if (cnt1 > cnt2)
                consensus_winners[m.id] = t1;
            else if (cnt2 > cnt1)
                consensus_winners[m.id] = t2;
            else {
                ProbInfo prob = get_prob(t1, t2);
                consensus_winners[m.id] = (prob.p_home_win >= prob.p_away_win) ? t1 : t2;
            }
        }
    }

    // 3rd place match
    int sf1 = 101, sf2 = 102;
    string loser1 = (consensus_winners[sf1] == resolve_slot(bracket[sf1-73].team1, consensus_winners))
                       ? resolve_slot(bracket[sf1-73].team2, consensus_winners)
                       : resolve_slot(bracket[sf1-73].team1, consensus_winners);
    string loser2 = (consensus_winners[sf2] == resolve_slot(bracket[sf2-73].team1, consensus_winners))
                       ? resolve_slot(bracket[sf2-73].team2, consensus_winners)
                       : resolve_slot(bracket[sf2-73].team1, consensus_winners);
    int g1, g2; string played;
    if (get_actual_result(loser1, loser2, g1, g2, played)) {
        consensus_winners[103] = played;
    } else {
        auto& counts = winner_counts[103];
        int cnt1 = counts[loser1];
        int cnt2 = counts[loser2];
        if (cnt1 > cnt2)
            consensus_winners[103] = loser1;
        else if (cnt2 > cnt1)
            consensus_winners[103] = loser2;
        else {
            ProbInfo prob = get_prob(loser1, loser2);
            consensus_winners[103] = (prob.p_home_win >= prob.p_away_win) ? loser1 : loser2;
        }
    }

    // ---------------------------------------------------------
    // 3. DISPLAY CONSENSUS BRACKET
    // ---------------------------------------------------------
    cout << "\n============================================================\n";
    cout << "  CONSENSUS BRACKET (most frequent winner at each stage)\n";
    cout << "============================================================\n";

    vector<MatchResult> det_results;
    unordered_map<int,string> consensus_losers;

    for (const auto& m : bracket) {
        if (m.phase == "3rd") continue;
        string t1 = resolve_slot(m.team1, consensus_winners);
        string t2 = resolve_slot(m.team2, consensus_winners);
        deterministic_display_match(m.id, m.phase, t1, t2, det_results);
        const auto& r = det_results.back();
        if (m.id == 101 || m.id == 102)
            consensus_losers[m.id] = r.loser;

        cout << left << setw(8) << ("[" + m.phase + "]")
             << "M" << setw(4) << m.id << ": "
             << setw(22) << t1 << " vs " << setw(22) << t2 << " → ";
        if (r.is_played) {
            cout << "Result: " << r.final_score << " (played)";
        } else {
            cout << "Prediction: " << r.final_score;
            if (r.extra_time) cout << " (AET)";
            if (r.penalties) cout << " (pens)";
        }
        cout << " [" << r.winner << " advances]\n";
    }

    // 3rd place
    string third_t1 = consensus_losers[101];
    string third_t2 = consensus_losers[102];
    deterministic_display_match(103, "3rd", third_t1, third_t2, det_results);
    const auto& r3 = det_results.back();
    cout << left << setw(8) << "[3rd]"
         << "M" << setw(4) << 103 << ": "
         << setw(22) << third_t1 << " vs " << setw(22) << third_t2 << " → ";
    if (r3.is_played) {
        cout << "Result: " << r3.final_score << " (played)";
    } else {
        cout << "Prediction: " << r3.final_score;
        if (r3.extra_time) cout << " (AET)";
        if (r3.penalties) cout << " (pens)";
    }
    cout << " [" << r3.winner << " wins 3rd place]\n";

    string champion = consensus_winners[104];
    string runner_up = (champion == resolve_slot(bracket.back().team1, consensus_winners))
                       ? resolve_slot(bracket.back().team2, consensus_winners)
                       : resolve_slot(bracket.back().team1, consensus_winners);
    string third_place = consensus_winners[103];
    string fourth_place = (third_place == third_t1) ? third_t2 : third_t1;

    cout << "\n--- Consensus Podium ---\n";
    cout << "  Champion:       " << champion << "\n";
    cout << "  Runner-up:      " << runner_up << "\n";
    cout << "  Third place:    " << third_place << "\n";
    cout << "  Fourth place:   " << fourth_place << "\n";

    // ---------------------------------------------------------
    // 4. PROBABILITY TABLE (alive teams)
    // ---------------------------------------------------------
    cout << "\n============================================================\n";
    cout << "  ADVANCEMENT PROBABILITIES (alive teams, " << SIMS << " simulations)\n";
    cout << "============================================================\n";

    cout << "\nTeam                     R16      QF       SF       Final    Champion\n";
    cout << "----------------------------------------------------------------------\n";
    vector<pair<string, array<int,5>>> alive_sorted;
    for (const auto& [team, s] : team_stats)
        if (!is_eliminated(team))
            alive_sorted.push_back({team, s});
    sort(alive_sorted.begin(), alive_sorted.end(),
         [](auto& a, auto& b) { return a.second[4] > b.second[4]; });

    for (const auto& [team, s] : alive_sorted) {
        cout << left << setw(25) << team;
        for (int i = 0; i < 5; ++i)
            cout << fixed << setprecision(1) << setw(9) << (100.0 * s[i] / SIMS);
        cout << "\n";
    }

    // ---------------------------------------------------------
    // 5. JSON OUTPUT
    // ---------------------------------------------------------
    ofstream json("../data/knockout_simulation_results.json");
    if (json.is_open()) {
        json << "{\n  \"consensus_bracket\": {\n";
        json << "    \"champion\": \"" << champion << "\",\n";
        json << "    \"runner_up\": \"" << runner_up << "\",\n";
        json << "    \"third\": \"" << third_place << "\",\n";
        json << "    \"fourth\": \"" << fourth_place << "\"\n  },\n  \"matches\": [\n";
        for (size_t i = 0; i < det_results.size(); ++i) {
            auto& m = det_results[i];
            json << "    {\"id\":" << m.id << ",\"phase\":\"" << m.phase
                 << "\",\"team1\":\"" << m.team1 << "\",\"team2\":\"" << m.team2
                 << "\",\"home_xg\":" << m.home_xg << ",\"away_xg\":" << m.away_xg
                 << ",\"is_played\":" << (m.is_played ? "true" : "false")
                 << ",\"90min_probs\":{\"t1\":" << m.t1_win_prob
                 << ",\"draw\":" << m.draw_prob << ",\"t2\":" << m.t2_win_prob
                 << "},\"score\":\"" << m.final_score << "\",\"ET\":" << (m.extra_time?"true":"false")
                 << ",\"pens\":" << (m.penalties?"true":"false")
                 << ",\"winner\":\"" << m.winner << "\",\"loser\":\"" << m.loser << "\"}"
                 << (i < det_results.size()-1 ? "," : "") << "\n";
        }
        json << "  ]\n}\n";
        json.close();
        cout << "\n✅ JSON saved to ../data/knockout_simulation_results.json\n";
    }

    return 0;
}