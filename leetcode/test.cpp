#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int value) : val(value), next(nullptr){}
};

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
        evenEnd->next = oddend->next;
        evenEnd = evenEnd->next;
    }
    oddend->next = evenStart;
    return head;
}
};

class Solution7 {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        vector<int> res;
        map<int, int> data;
        for (auto arr : arr1) {
            data[arr]++;
        }
        for (auto arr :arr2) {
            if (data.count(arr) > 0) {
                for (int i = 0; i < data[arr]; i++) {
                    res.push_back(arr);
                }
                data.erase(arr);
            }
        }

        for (auto iter = data.begin(); iter != data.end(); iter++) {
            for (int i = 0; i < iter->second; i++) {
                res.push_back(iter->first);
            }
        }
        random_shuffle(res.begin(), res.end());
        return res;

    }
};

class Solution8 {
public:
    string removeKdigits(string num, int k) {
        vector<char> stk;
        for (auto& digit: num) {
            while (stk.size() > 0 && stk.back() > digit && k) {
                stk.pop_back();
                k -= 1;
            }
            stk.push_back(digit);
        }

        for (; k > 0; --k) {
            stk.pop_back();
        }

        string ans = "";
        bool isLeadingZero = true;
        for (auto& digit: stk) {
            if (isLeadingZero && digit == '0') {
                continue;
            }
            isLeadingZero = false;
            ans += digit;
        }
        return ans == "" ? "0" : ans;
    }
};

class Solution9 {
public:
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        if(people.size() == 0) return {};
        // 先把数组按照身高 从高到低排序
        sort(people.begin(), people.end(), [](vector<int>& a, vector<int>& b)
        {
            return a[0] > b[0] ||(a[0] == b[0] && a[1] < b[1]);
        });
        vector<vector<int>> res;

        // 然后重新开个数组  按顺序把 它插入到数组中的k的位置上
        for(auto a: people)
        {
            res.insert(res.begin() + a[1], a);
        }
        return res;
    }
};

class Solution10 {
public:
    const int dr[4] = {1, 1, -1, -1};
    const int dc[4] = {1, -1, -1, 1};

    vector<vector<int>> allCellsDistOrder(int R, int C, int r0, int c0) {
        int maxDist = max(r0, R - 1 - r0) + max(c0, C - 1 - c0);
        vector<vector<int>> ret;
        int row = r0, col = c0;
        ret.push_back({row, col});
        for (int dist = 1; dist <= maxDist; dist++) {
            row--;
            for (int i = 0; i < 4; i++) {
                while ((i % 2 == 0 && row != r0) || (i % 2 != 0 && col != c0)) {
                    if (row >= 0 && row < R && col >= 0 && col < C) {
                        ret.push_back({row, col});
                    }
                    row += dr[i];
                    col += dc[i];
                }
            }
        }
        return ret;
    }
};

class Solution11 {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        int i = 0;
        while (i < n) {
            int sumOfGas = 0, sumOfCost = 0;
            int cnt = 0;
            while (cnt < n) {
                int j = (i + cnt) % n;
                sumOfGas += gas[j];
                sumOfCost += cost[j];
                if (sumOfCost > sumOfGas) {
                    break;
                }
                cnt++;
            }
            if (cnt == n) {
                return i;
            } else {
                i = i + cnt + 1;
            }
        }
        return -1;
    }
};

int main() {

    vector<int> arr2 = {2,3,1,5,6};
    vector<int> arr1 = {3,7,8,6,5,4,3,1,2,2,3,4,65,7,5,3,2,4,5};
    vector<int> res;
    Solution7 s;
    res = s.relativeSortArray(arr1, arr2);
    for (auto a : res) {
        cout<<a<<endl;
    }
    return 0;
}