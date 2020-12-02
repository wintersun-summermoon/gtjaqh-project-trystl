#include "test.h"
#include <iostream>
#include <vector>
#include <string>
#include <string.h>
#include <map>
#include <deque>
#include <queue>
#include <set>
#include <unordered_map>
#include <algorithm>
#include<functional>

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

class Solution12 {
public:
    ListNode* insertionSortList(ListNode* head) {
        if (head == nullptr) {
            return head;
        }
        ListNode* dummyHead = new ListNode(0);
        dummyHead->next = head;
        ListNode* lastSorted = head;
        ListNode* curr = head->next;
        while (curr != nullptr) {
            if (lastSorted->val <= curr->val) {
                lastSorted = lastSorted->next;
            } else {
                ListNode *prev = dummyHead;
                while (prev->next->val <= curr->val) {
                    prev = prev->next;
                }
                lastSorted->next = curr->next;
                curr->next = prev->next;
                prev->next = curr;
            }
            curr = lastSorted->next;
        }
        return dummyHead->next;
    }
};

class Solution13 {
public:
    int maximumGap(vector<int>& nums) {
        int n = nums.size();
        if (n < 2) {
            return 0;
        }
        int minVal = *min_element(nums.begin(), nums.end());
        int maxVal = *max_element(nums.begin(), nums.end());
        int d = max(1, (maxVal - minVal) / (n - 1));
        int bucketSize = (maxVal - minVal) / d + 1;

        vector<pair<int, int>> bucket(bucketSize, {-1, -1});  // 存储 (桶内最小值，桶内最大值) 对，(-1, -1) 表示该桶是空的
        for (int i = 0; i < n; i++) {
            int idx = (nums[i] - minVal) / d;
            if (bucket[idx].first == -1) {
                bucket[idx].first = bucket[idx].second = nums[i];
            } else {
                bucket[idx].first = min(bucket[idx].first, nums[i]);
                bucket[idx].second = max(bucket[idx].second, nums[i]);
            }
        }

        int ret = 0;
        int prev = -1;
        for (int i = 0; i < bucketSize; i++) {
            if (bucket[i].first == -1) continue;
            if (prev != -1) {
                ret = max(ret, bucket[i].first - bucket[prev].second);
            }
            prev = i;
        }
        return ret;
    }
};

class Solution14 {
public:
    void moveZeroes(vector<int>& nums) {
        int n = nums.size(), left = 0, right = 0;
        while (right < n) {
            if (nums[right]) {
                swap(nums[left], nums[right]);
                left++;
            }
            right++;
        }
    }
};

class Solution15 {
public:
    ListNode* sortList(ListNode* head) {
        return sortList(head, nullptr);
    }

    ListNode* sortList(ListNode* head, ListNode* tail) {
        if (head == nullptr) {
            return head;
        }
        if (head->next == tail) {
            head->next = nullptr;
            return head;
        }
        ListNode* slow = head, *fast = head;
        while (fast != tail) {
            slow = slow->next;
            fast = fast->next;
            if (fast != tail) {
                fast = fast->next;
            }
        }
        ListNode* mid = slow;
        return merge(sortList(head, mid), sortList(mid, tail));
    }

    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode* dummyHead = new ListNode(0);
        ListNode* temp = dummyHead, *temp1 = head1, *temp2 = head2;
        while (temp1 != nullptr && temp2 != nullptr) {
            if (temp1->val <= temp2->val) {
                temp->next = temp1;
                temp1 = temp1->next;
            } else {
                temp->next = temp2;
                temp2 = temp2->next;
            }
            temp = temp->next;
        }
        if (temp1 != nullptr) {
            temp->next = temp1;
        } else if (temp2 != nullptr) {
            temp->next = temp2;
        }
        return dummyHead->next;
    }
};

class Solution16 {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        int res = 1;
        sort(points.begin(), points.end(),[&](vector<int>&a, vector<int>&b){
            return a[1] < b[1];
        });
        int right = points[0][1];
        for (int i = 1; i < points.size(); i++) {
            if (points[i][0] > right) {
                right = points[i][1];
                res++;
            }
        }
        return res;
    }
};

class Solution17 {
public:
    int countNodes(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        int level = 0;
        TreeNode* node = root;
        while (node->left != nullptr) {
            level++;
            node = node->left;
        }
        int low = 1 << level, high = (1 << (level + 1)) - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (exists(root, level, mid)) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    bool exists(TreeNode* root, int level, int k) {
        int bits = 1 << (level - 1);
        TreeNode* node = root;
        while (node != nullptr && bits > 0) {
            if (!(bits & k)) {
                node = node->left;
            } else {
                node = node->right;
            }
            bits >>= 1;
        }
        return node != nullptr;
    }
};

class Solution18 {
public:
    string sortString(string s) {
        vector<int> num(26);
        for (char &ch : s) {
            num[ch - 'a']++;
        }

        string ret;
        while (ret.length() < s.length()) {
            for (int i = 0; i < 26; i++) {
                if (num[i]) {
                    ret.push_back(i + 'a');
                    num[i]--;
                }
            }
            for (int i = 25; i >= 0; i--) {
                if (num[i]) {
                    ret.push_back(i + 'a');
                    num[i]--;
                }
            }
        }
        return ret;
    }
};

class Solution19 {
public:
int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
    int size = A.size();
    if (size == 0) return 0;
    unordered_map<int, int> result;
    int res = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result[A[i]+B[j]]++;
        }
    }
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++) {
            if (result.find(0-C[i]-D[j])!= result.end()) {
                res+= result[0-C[i]-D[j]];
            }
        }
    }
    return res;
}
};

class Solution20 {
public:
    int reversePairsRecursive(vector<int>& nums, int left, int right) {
        if (left == right) {
            return 0;
        } else {
            int mid = (left + right) / 2;
            int n1 = reversePairsRecursive(nums, left, mid);
            int n2 = reversePairsRecursive(nums, mid + 1, right);
            int ret = n1 + n2;

            // 首先统计下标对的数量
            int i = left;
            int j = mid + 1;
            while (i <= mid) {
                while (j <= right && (long long)nums[i] > 2 * (long long)nums[j]) j++;
                ret += (j - mid - 1);
                i++;
            }

            // 随后合并两个排序数组
            vector<int> sorted(right - left + 1);
            int p1 = left, p2 = mid + 1;
            int p = 0;
            while (p1 <= mid || p2 <= right) {
                if (p1 > mid) {
                    sorted[p++] = nums[p2++];
                } else if (p2 > right) {
                    sorted[p++] = nums[p1++];
                } else {
                    if (nums[p1] < nums[p2]) {
                        sorted[p++] = nums[p1++];
                    } else {
                        sorted[p++] = nums[p2++];
                    }
                }
            }
            for (int i = 0; i < sorted.size(); i++) {
                nums[left + i] = sorted[i];
            }
            return ret;
        }
    }

    int reversePairs(vector<int>& nums) {
        if (nums.size() == 0) return 0;
        return reversePairsRecursive(nums, 0, nums.size() - 1);
    }
};

class Solution21 {
public:
    int largestPerimeter(vector<int>& A) {
        int size = A.size();
        if (size < 3) return 0;
        sort(A.begin(), A.end());
        for (int i = size-1; i >=2; i--) {
            if (A[i-1]+A[i-2] > A[i]) return A[i]+A[i-1]+A[i-2];
        }
        return 0;
    }
};

class Solution22 {
public:
    string reorganizeString(string S) {
        if (S.length() < 2) {
            return S;
        }
        vector<int> counts(26, 0);
        int maxCount = 0;
        int length = S.length();
        for (int i = 0; i < length; i++) {
            char c = S[i];
            counts[c - 'a']++;
            maxCount = max(maxCount, counts[c - 'a']);
        }
        if (maxCount > (length + 1) / 2) {
            return "";
        }
        auto cmp = [&](const char& letter1, const char& letter2) {
            return counts[letter1 - 'a']  < counts[letter2 - 'a'];
        };
        priority_queue<char, vector<char>,  decltype(cmp)> queue{cmp};
        for (char c = 'a'; c <= 'z'; c++) {
            if (counts[c - 'a'] > 0) {
                queue.push(c);
            }
        }
        string sb = "";
        while (queue.size() > 1) {
            char letter1 = queue.top(); queue.pop();
            char letter2 = queue.top(); queue.pop();
            sb += letter1;
            sb += letter2;
            int index1 = letter1 - 'a', index2 = letter2 - 'a';
            counts[index1]--;
            counts[index2]--;
            if (counts[index1] > 0) {
                queue.push(letter1);
            }
            if (counts[index2] > 0) {
                queue.push(letter2);
            }
        }
        if (queue.size() > 0) {
            sb += queue.top();
        }
        return sb;
    }
};


class Solution {
public:
    vector<int> maxNumber(vector<int>& nums1, vector<int>& nums2, int k) {
        int m = nums1.size(), n = nums2.size();
        vector<int> maxSubsequence(k, 0);
        int start = max(0, k - n), end = min(k, m);
        for (int i = start; i <= end; i++) {
            vector<int> subsequence1(MaxSubsequence(nums1, i));
            vector<int> subsequence2(MaxSubsequence(nums2, k - i));
            vector<int> curMaxSubsequence(merge(subsequence1, subsequence2));
            if (compare(curMaxSubsequence, 0, maxSubsequence, 0) > 0) {
                maxSubsequence.swap(curMaxSubsequence);
            }
        }
        return maxSubsequence;
    }

    vector<int> MaxSubsequence(vector<int>& nums, int k) {
        int length = nums.size();
        vector<int> stack(k, 0);
        int top = -1;
        int remain = length - k;
        for (int i = 0; i < length; i++) {
            int num = nums[i];
            while (top >= 0 && stack[top] < num && remain > 0) {
                top--;
                remain--;
            }
            if (top < k - 1) {
                stack[++top] = num;
            } else {
                remain--;
            }
        }
        return stack;
    }

    vector<int> merge(vector<int>& subsequence1, vector<int>& subsequence2) {
        int x = subsequence1.size(), y = subsequence2.size();
        if (x == 0) {
            return subsequence2;
        }
        if (y == 0) {
            return subsequence1;
        }
        int mergeLength = x + y;
        vector<int> merged(mergeLength);
        int index1 = 0, index2 = 0;
        for (int i = 0; i < mergeLength; i++) {
            if (compare(subsequence1, index1, subsequence2, index2) > 0) {
                merged[i] = subsequence1[index1++];
            } else {
                merged[i] = subsequence2[index2++];
            }
        }
        return merged;
    }

    int compare(vector<int>& subsequence1, int index1, vector<int>& subsequence2, int index2) {
        int x = subsequence1.size(), y = subsequence2.size();
        while (index1 < x && index2 < y) {
            int difference = subsequence1[index1] - subsequence2[index2];
            if (difference != 0) {
                return difference;
            }
            index1++;
            index2++;
        }
        return (x - index1) - (y - index2);
    }
};

int main() {
    return 0;
}