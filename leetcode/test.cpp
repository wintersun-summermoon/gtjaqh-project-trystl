#include <vector>
#include <string>
#include <string.h>
#include <map>
#include <queue>
#include <algorithm>

using namespace std;

class Solution1 {
public:
    vector<int> sortByBits(vector<int>& arr) {
        vector<int> bit(10001, 0);
        for (int i = 1;i <= 10000; ++i) {
            bit[i] = bit[i>>1] + (i & 1);
        }
        sort(arr.begin(),arr.end(),[&](int x,int y){
            if (bit[x] < bit[y]) {
                return true;
            }
            if (bit[x] > bit[y]) {
                return false;
            }
            return x < y;
        });
        return arr;
    }
};

class Solution2 {
public:
    vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
        vector<vector<int>> res;
        if (points.size() == 0) return res;
        map<int, vector<vector<int>>> data;
        for (auto point : points) {
            int length = point[0]*point[0] + point[1]*point[1];
            data[length].push_back(point);
        }
        auto iter = data.begin();
        while(K>0) {
            K-=iter->second.size();
            for (auto t : iter->second) {
                res.push_back(t);
            }
            iter++;
        }
        return res;

    }
};

class Solution3 {
public:
   void reverse(vector<int>& nums, int k){
    int i = k;
    int j  = nums.size()-1;
    while(i<j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
        i++;
        j--;
    }
}

void nextPermutation(vector<int>& nums) {
    int size = nums.size();
    if (size<2) return;
    int k = -1;
    for (int i = size-1; i >0; i--) {
        if (nums[i]>nums[i-1]) {
            k = i-1;
            break;
        }
    }
    if (k==-1) {
        reverse(nums, 0);
    } else {
        for (int i = size-1; i>=k;i--) {
            if (nums[i]>nums[k]) {
                int temp = nums[i];
                nums[i] = nums[k];
                nums[k] = temp;
                reverse(nums, k+1);
                break;
            }
        }
    }
    return;

}
};

class State {
    public:
        int step;
        int cur;
        int res;
        State (int step, int cur, int res) {
            this->step = step;
            this->cur = cur;
            this->res = res;
        }
        bool operator < (const State &s) const {
            return step > s.step;
        }
};

class Solution4 {
public:
    int findRotateSteps(string ring, string key) {
        // 思路：优先队列实现 bfs
        int n = ring.size();
        int m = key.size();
        vector<vector<int>> pos(27, vector<int>());
        vector<vector<int>> memo(n + 1, vector<int>(m + 1, INT_MAX));

        for (int i = 0; i < n; i++) {
            pos[ring[i] - 'a'].push_back(i);
        }

        if (n == 0 || m == 0)
            return 0;
        
        priority_queue<State> PQ;

        for (const int& idx : pos[key[0] - 'a']) {
            PQ.push(State(min(idx, n - idx) + 1, idx, 1));
        }

        while (!PQ.empty()) {
            State top = PQ.top(); PQ.pop();

            if (top.res >= m) {
                return top.step;
            }

            for (const int& idx : pos[key[top.res] - 'a']) {
                int dist = abs(top.cur - idx);
                int step = top.step + min(dist, n - dist) + 1;
                if (step < memo[idx][top.res + 1]) {
                    memo[idx][top.res + 1] = step;
                    PQ.push(State(step, idx, top.res + 1));
                }
            }
        }

        return 0;
    }
};

class Solution5 {
public:
    int findRotateSteps(string ring, string key) {
        int n = ring.size(), m = key.size();
        vector<int> pos[26];
        for (int i = 0; i < n; ++i) {
            pos[ring[i] - 'a'].push_back(i);
        }
        int dp[m][n];
        memset(dp, 0x3f3f3f3f, sizeof(dp));
        for (auto& i: pos[key[0] - 'a']) {
            dp[0][i] = min(i, n - i) + 1;
        }
        for (int i = 1; i < m; ++i) {
            for (auto& j: pos[key[i] - 'a']) {
                for (auto& k: pos[key[i - 1] - 'a']) {
                    dp[i][j] = min(dp[i][j], dp[i - 1][k] + min(abs(j - k), n - abs(j - k)) + 1);
                }
            }
        }
        return *min_element(dp[m - 1], dp[m - 1] + n);
    }
};

class Solution6 {
public:
ListNode* oddEvenList(ListNode* head) {
    if (!head || !(head->next) || !(head->next->next)) return head;
    ListNode *oddend, *evenStart, *evenEnd;
    oddend = head;
    evenStart = evenEnd = head->next;
    while (evenEnd &&evenEnd->next) {
        oddend->next = evenEnd->next;
        oddend = oddend->next;
        evenEnd->next = oddend->next;;
        evenEnd = evenEnd->next;
    }
    oddend->next = evenStart;
    return head;
}
};

int main() {
    Solution6 aa;
    aa.findRotateSteps("asdfgh", "ah");
    return 0;
}