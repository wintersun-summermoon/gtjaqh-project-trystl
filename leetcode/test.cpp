#include <vector>
#include <string>
#include <map>
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

int main() {
    return 0;
}