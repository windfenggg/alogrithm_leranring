//448. 找到所有数组中消失的数字
////哈希表
//时间复杂度：O(n)
//空间复杂度：O(n)
//
//class Solution {
//public:
//    vector<int> findDisappearedNumbers(vector<int>& nums) {
//        unordered_set<int>se;
//        for (auto& i : nums) {
//            se.insert(i);
//        }
//        vector<int> ans;
//        for (int i = 1; i <= nums.size(); i++) {
//            if (se.find(i) == se.end()) {
//                ans.push_back(i);
//            }
//        }
//        return ans;
//    }
//};


//优化
//哈希+取模
//将存在的数字都+n
//当我们遍历到某个位置时，其中的数可能已经被增加过，因此需要对 n 取模来还原出它本来的值
//时间复杂度：O(n)
//空间复杂度：O(1)
/*#include<iostream>
#include<vector>
using namespace std;

class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        for (auto& num : nums) {
            int x = (num - 1) % n;
            nums[x] += n;
        }
        vector<int> ret;
        for (int i = 0; i < n; i++) {
            if (nums[i] <= n) {
                ret.push_back(i + 1);
            }
        }
        return ret;
    }
};

int main() {
    Solution s;
    vector<int>v = { 4,3,2,7,8,2,3,1 };
    vector<int>ans=s.findDisappearedNumbers(v);
    for (int i = 0; i < ans.size(); i++) {
        cout << ans[i] << endl;
    }
    return  0;
}*/


//3. 无重复字符的最长子串
//给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
/*class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int>um;
        int ans = 0;
        for (int left = 0, right = 0; right < s.size(); right++) {
            um[s[right]]++;
            while (um[s[right]] > 1) {
                um[s[left]]--;
                left++;
            }
            ans = max(ans, right - left + 1);
        }
        return ans;
    }
};*/



//11. 盛最多水的容器
//给定一个长度为 n 的整数数组 height 。有 n 条垂线，第 i 条线的两个端点是(i, 0) 和(i, height[i]) 。
//找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。
//返回容器可以储存的最大水量。
//说明：你不能倾斜容器。

//暴力超时
/*class Solution {
public:
    int maxArea(vector<int>& height) {
        int n = height.size();
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                ans = max(ans, (min(height[j], height[i])) * (j - i));
            }
        }
        return ans;
    }
};*/

//时间复杂度：O(n)
//空间复杂度：O(1)
//双指针
/*
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i = 0, j = height.size() - 1, res = 0;
        while (i < j) {
            res = height[i] < height[j] ?
                max(res, (j - i) * height[i++]) :
                max(res, (j - i) * height[j--]);
        }
        return res;
    }
};*/

//22. 括号生成
//数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合
//方法一：暴力法
/*#include<iostream>
#include<vector>
#include<string>
using namespace std;

class Solution {
    bool valid(const string& str) {
        int balance = 0;
        for (char c : str) {
            if (c == '(') {
                ++balance;
            }
            else {
                --balance;
            }
            if (balance < 0) {
                return false;
            }
        }
        return balance == 0;
    }

    void generate_all(string& current, int n, vector<string>& result) {
        if (n == current.size()) {
            if (valid(current)) {
                result.push_back(current);
            }
            return;
        }
        current += '(';
        generate_all(current, n, result);
        current.pop_back();
        current += ')';
        generate_all(current, n, result);
        current.pop_back();
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string current;
        generate_all(current, n * 2, result);
        return result;
    }
};*/

//优化
/*class Solution {
    void backtrack(vector<string>& ans, string& cur, int open, int close, int n) {
        if (cur.size() == n * 2) {
            ans.push_back(cur);
            return;
        }
        if (open < n) {
            cur.push_back('(');
            backtrack(ans, cur, open + 1, close, n);
            cur.pop_back();
        }
        if (close < open) {
            cur.push_back(')');
            backtrack(ans, cur, open, close + 1, n);
            cur.pop_back();
        }
    }
public:
    vector<string> generateParenthesis(int n) {
        vector<string> result;
        string current;
        backtrack(result, current, 0, 0, n);
        return result;
    }
};*/


//33. 搜索旋转排序数组
//暴力搜索
//时间复杂度：O(n)
//空间复杂度：O(1)
/*
class Solution {
public:
    int search(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] == target) {
                return i;
            }
        }
        return -1;
    }
};*/


//时间复杂度：O(logN)
//空间复杂度：O(1)
//二分法，分为2个有序区间，然后二分
/*class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = (left + right) >> 1;
            if (nums[mid] == target) return mid;
            if (nums[left] <= nums[mid]) {
                (target >= nums[left] && target < nums[mid]) ? right = mid - 1 : left = mid + 1;
            }
            else {
                (target > nums[mid] && target <= nums[right]) ? left = mid + 1 : right = mid - 1;
            }
        }
        return -1;
   
};*/


//48. 旋转图像
// 法一，辅助数组
//对于矩阵中第 i 行的第j 个元素，在旋转后，它出现在倒数第 i列的第j 个位置。
//时间复杂度：O(n2)
//空间复杂度：O(n2)
/*class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // C++ 这里的 = 拷贝是值拷贝，会得到一个新的数组
        auto matrix_new = matrix;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_new[j][n - i - 1] = matrix[i][j];
            }
        }
        // 这里也是值拷贝
        matrix = matrix_new;
    }
};*/


//法二，原地旋转，临时变量
//时间复杂度：O(n2)
//空间复杂度：O(1)
/*
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < (n + 1) / 2; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }
};*/


//水平翻转，主对角线翻转
//时间复杂度：O(n2)
//空间复杂度：O(1)
/*class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 水平翻转
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < n; ++j) {
                swap(matrix[i][j], matrix[n - i - 1][j]);
            }
        }
        // 主对角线翻转
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
    }
};*/



//49. 字母异位词分组
//使用哈希表存储每一组字母异位词，哈希表的键为一组字母异位词的标志，
//哈希表的值为一组字母异位词列表。
//遍历每个字符串，对于每个字符串，得到该字符串所在的一组字母异位词的标志
//将当前字符串加入该组字母异位词的列表中。遍历全部字符串之后，
//哈希表中的每个键值对即为一组字母异位词。
//时间复杂度：O(nklogk),k为str长度
//空间复杂度：O(nk)
/*
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string& str : strs) {
            string key = str;
            sort(key.begin(), key.end());
            mp[key].emplace_back(str);
        }
        vector<vector<string>> ans;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            ans.emplace_back(it->second);
        }
        return ans;
    }
};*/


//79. 单词搜索
//回溯算法
//s[k]表示字符串s是从第k个字符开始的后缀子串
/*
class Solution {
public:
    vector<pair<int, int>>directions{ {-1,0},{1,0},{0,1},{0,-1} };
    bool dfs(vector<vector<char>>& board, vector<vector<int>>& visited, int i, int j, string& s, int k) {
        if (board[i][j] != s[k]) {
            return false;
        }
        else if (k == s.size() - 1) {
            return true;
        }
        visited[i][j] = true;
        bool result = false;
        for (const auto& dir : directions) {
            int newi = i + dir.first, newj = j + dir.second;
            if (newi >= 0 && newi < board.size() && newj >= 0 && newj < board[0].size()) {
                if (!visited[newi][newj]) {
                    if (dfs(board, visited, newi, newj, s, k + 1)) {
                        result = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;
        return result;
    }

    bool exist(vector<vector<char>>& board, string word) {
        vector<vector<int>>visited(board.size(), vector<int>(board[0].size(), 0));
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                if (dfs(board, visited, i, j, word, 0)) {
                    return true;
                }
            }
        }
        return false;
    }
};*/


//哈希表 + 双向链表
//自己设计链表
//LRU缓存机制： Least Recently Used，最近使用过的数据应该是有用的
//很久都没用过的数据应该是无用的，内存满了就优先删那些很久没用过的数据。
/*struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode() : key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value) : key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

public:
    LRUCache(int _capacity) : capacity(_capacity), size(0) {
        // 使用伪头部和伪尾部节点
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (!cache.count(key)) {
            return -1;
        }
        // 如果 key 存在，先通过哈希表定位，再移到头部
        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }

    void put(int key, int value) {
        if (!cache.count(key)) {
            // 如果 key 不存在，创建一个新的节点
            DLinkedNode* node = new DLinkedNode(key, value);
            // 添加进哈希表
            cache[key] = node;
            // 添加至双向链表的头部
            addToHead(node);
            ++size;
            if (size > capacity) {
                // 如果超出容量，删除双向链表的尾部节点
                DLinkedNode* removed = removeTail();
                // 删除哈希表中对应的项
                cache.erase(removed->key);
                // 防止内存泄漏
                delete removed;
                --size;
            }
        }
        else {
            // 如果 key 存在，先通过哈希表定位，再修改 value，并移到头部
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }

    void addToHead(DLinkedNode* node) {
        node->prev = head;
        node->next = head->next;
        head->next->prev = node;
        head->next = node;
    }

    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* removeTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);
        return node;
    }
};*/



//114. 二叉树展开为链表
//递归实现前序遍历
//时间复杂度O(n)
//空间复杂度O(n)
/*
class Solution {
public:
    vector<TreeNode*>ans;
    void dfs(TreeNode* root) {
        if (root == NULL) {
            return;
        }
        ans.push_back(root);
        dfs(root->left);
        dfs(root->right);
    }

    void flatten(TreeNode* root) {
        dfs(root);
        for (int i = 1; i < ans.size(); i++) {
            TreeNode* prev = ans[i - 1];
            TreeNode* curr = ans[i];
            prev->left = NULL;
            prev->right = curr;
        }
    }
};*/

//法2递归
//其实是分为三步：
//1）首先将根节点的左子树变成链表
//2）其次将根节点的右子树变成链表
//3）最后将变成链表的右子树放在变成链表的左子树的最右边
//这就是一个递归的过程，递归的一个非常重要的点就是：、
//不去管函数的内部细节是如何处理的，我们只看其函数作用以及输入与输出。
//对于函数flatten来说：
//函数作用：将一个二叉树，原地将它展开为链表
//输入：树的根节点
//输出：无


/*class Solution {
    void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        //将根节点的左子树变成链表
        flatten(root.left);
        //将根节点的右子树变成链表
        flatten(root.right);
        TreeNode temp = root.right;
        //把树的右边换成左边的链表
        root.right = root.left;
        //记得要将左边置空
        root.left = null;
        //找到树的最右边的节点
        while (root.right != null) root = root.right;
        //把右边的链表接到刚才树的最右边的节点
        root.right = temp;
    }
}*/


//法三，寻找前驱节点
//时间复杂度O(n)
//空间复杂度O(1)
/*
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* curr = root;
        while (curr != nullptr) {
            if (curr->left != nullptr) {
                auto next = curr->left;
                auto predecessor = next;
                while (predecessor->right != nullptr) {
                    predecessor = predecessor->right;
                }
                predecessor->right = curr->right;
                curr->left = nullptr;
                curr->right = next;
            }
            curr = curr->right;
        }
    }
};*/


//148. 排序链表
//法一。转换成数组排序再放回链表
//时间复杂度O(nlogn)
//空间复杂度O(n)
/*
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (!head) return head;
        vector<int> a;
        auto p = head;
        while (p != NULL) {
            a.push_back(p->val);
            p = p->next;
        }
        delete p;
        sort(a.begin(), a.end());	//可以自己写个快排，会提升不少效率
        auto q = head;
        for (const auto& c : a) {
            q->val = c;
            q = q->next;
        }
        delete q;

        return head;
    }
};*/


//归并排序
//时间复杂度O(nlogn)
//空间复杂度O(logn)
//找链表的中点，快慢指针，合并两个有序链表
/*class Solution {
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
        ListNode* slow = head, * fast = head;
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
        ListNode* temp = dummyHead, * temp1 = head1, * temp2 = head2;
        while (temp1 != nullptr && temp2 != nullptr) {
            if (temp1->val <= temp2->val) {
                temp->next = temp1;
                temp1 = temp1->next;
            }
            else {
                temp->next = temp2;
                temp2 = temp2->next;
            }
            temp = temp->next;
        }
        if (temp1 != nullptr) {
            temp->next = temp1;
        }
        else if (temp2 != nullptr) {
            temp->next = temp2;
        }
        return dummyHead->next;
    }
};*/



//152. 乘积最大子数组
//动态规划
//因为有正负，维护最大值和最小值
//1）dp[i]:以i为下标的最大子数组乘积
//2）递推公式：dp[i]=max(dp[i-1]*nums[i],dp2[i-1]*nums[i],nums[i])
//  dp2[i]=min(dp[i-1]*nums[i],dp[i-1]*nums[i],nums[i])
//3)初始化
//4）从前往后
//5）举例
/*
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        vector <int> maxF(nums), minF(nums);
        for (int i = 1; i < nums.size(); ++i) {
            maxF[i] = max(maxF[i - 1] * nums[i], max(nums[i], minF[i - 1] * nums[i]));
            minF[i] = min(minF[i - 1] * nums[i], min(nums[i], maxF[i - 1] * nums[i]));
        }
        return *max_element(maxF.begin(), maxF.end());
    }
};*/



//200. 岛屿数量

//二叉树DFS
/*void traverse(TreeNode* root) {
    // 判断 base case
    if (root == NULL) {
        return;
    }
    // 访问两个相邻结点：左子结点、右子结点
    traverse(root.left);
    traverse(root.right);
}*/

//网格DFS，有4个方向
//发现 root == null 再返回和发现数组越界返回一样
/*void dfs(vector<vector<int>>grid, int r, int c) {
    // 判断 base case
    // 如果坐标 (r, c) 超出了网格范围，直接返回
    if (!inArea(grid, r, c)) {
        return;
    }
    // 访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
bool inArea(vector<vector<int>>grid, int r, int c) {
    return 0 <= r && r < grid.length
        && 0 <= c && c < grid[0].length;
}*/


//避免重复遍历
//0 —— 海洋格子
//1 —— 陆地格子（未遍历过）
//2 —— 陆地格子（已遍历过）
/*void dfs(vector<vector<int>>grid, int r, int c) {
    // 判断 base case
    if (!inArea(grid, r, c)) {
        return;
    }
    // 如果这个格子不是岛屿，直接返回
    if (grid[r][c] != 1) {
        return;
    }
    grid[r][c] = 2; // 将格子标记为「已遍历过」

    //访问上、下、左、右四个相邻结点
    dfs(grid, r - 1, c);
    dfs(grid, r + 1, c);
    dfs(grid, r, c - 1);
    dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
 bool inArea(vector<vector<int>>grid, int r, int c) {
    return 0 <= r && r < grid.length
        && 0 <= c && c < grid[0].length;
}*/


//模板题解
//其实就是一个递归标注的过程，它会将所有相连的1都标注成2
/*class Solution {
private:
    void dfs(vector<vector<char>>& grid, int i, int j) {
        //把上面的inArea函数和格子不是岛屿一起写了
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] != '1') {
            return;
        }
        grid[i][j] = '2';
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }

public:
    int numIslands(vector<vector<char>>& grid) {
        int num = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    num++;
                }
            }
        }
        return num;
    }
};*/



//官方题解
/*class Solution {
private:
    void dfs(vector<vector<char>>& grid, int r, int c) {
        int nr = grid.size();
        int nc = grid[0].size();

        grid[r][c] = '0';
        if (r - 1 >= 0 && grid[r - 1][c] == '1') dfs(grid, r - 1, c);
        if (r + 1 < nr && grid[r + 1][c] == '1') dfs(grid, r + 1, c);
        if (c - 1 >= 0 && grid[r][c - 1] == '1') dfs(grid, r, c - 1);
        if (c + 1 < nc && grid[r][c + 1] == '1') dfs(grid, r, c + 1);
    }

public:
    int numIslands(vector<vector<char>>& grid) {
        int nr = grid.size();
        if (!nr) return 0;
        int nc = grid[0].size();

        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }
};*/



//463. 岛屿的周长
//当我们的 dfs 函数因为「坐标 (r, c) 超出网格范围」返回的时候，黄色的边是与网格边界相邻的周长，
//当函数因为「当前格子是海洋格子」返回的时候，蓝色的边是与海洋格子相邻的周长。
/*int islandPerimeter(vector<vector<int>>& grid) {
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            if (grid[r][c] == 1) {
                // 题目限制只有一个岛屿，计算一个即可
                return dfs(grid, r, c);
            }
        }
    }
    return 0;
}

int dfs(vector<vector<int>>& grid, int r, int c) {
    // 函数因为「坐标 (r, c) 超出网格范围」返回，对应一条黄色的边
    if (!inArea(grid, r, c)) {
        return 1;
    }
    // 函数因为「当前格子是海洋格子」返回，对应一条蓝色的边
    if (grid[r][c] == 0) {
        return 1;
    }
    // 函数因为「当前格子是已遍历的陆地格子」返回，和周长没关系
    if (grid[r][c] != 1) {
        return 0;
    }
    grid[r][c] = 2;
    return dfs(grid, r - 1, c)
        + dfs(grid, r + 1, c)
        + dfs(grid, r, c - 1)
        + dfs(grid, r, c + 1);
}

// 判断坐标 (r, c) 是否在网格中
bool inArea(vector<vector<int>>& grid, int r, int c) {
    return 0 <= r && r < grid.length
        && 0 <= c && c < grid[0].length;
}*/


//数学方法
//总周长 = 4 * 土地个数 - 2 * 接壤边的条数。
/*
class Solution {
public:
    int islandPerimeter(vector<vector<int>>& grid) {
        int sum = 0;//土地个数
        int count = 0;//相邻陆地
        for (int row = 0; row < grid.size(); row++) {
            for (int col = 0; col < grid[0].size(); col++) {
                if (grid[row][col] == 1) {
                    sum++;
                    if (row >= 1 && grid[row - 1][col] == 1) count++;
                    if (col >= 1 && grid[row][col - 1]) count++;
                }
            }
        }
        return 4 * sum - 2 * count;

    }
};*/


//695. 岛屿的最大面积
//法一模板
/*int maxAreaOfIsland(vector<vector<int>>& grid) {
    int res = 0;
    for (int r = 0; r < grid.length; r++) {
        for (int c = 0; c < grid[0].length; c++) {
            if (grid[r][c] == 1) {
                int a = area(grid, r, c);
                res = Math.max(res, a);
            }
        }
    }
    return res;
}

int area(vector<vector<int>>& grid, int r, int c) {
    if (!inArea(grid, r, c)) {
        return 0;
    }
    if (grid[r][c] != 1) {
        return 0;
    }
    grid[r][c] = 2;

    return 1
        + area(grid, r - 1, c)
        + area(grid, r + 1, c)
        + area(grid, r, c + 1);
        + area(grid, r, c - 1)
}

bool inArea(vector<vector<int>>& grid, int r, int c) {
    return 0 <= r && r < grid.length
        && 0 <= c && c < grid[0].length;
}*/


//法二简洁版
/*
class Solution {
private:
    int dfs(vector<vector<int>>& grid, int i, int j) {
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] != 1) {
            return 0;
        }
        grid[i][j] = 2;
        return 1 + dfs(grid, i - 1, j) + dfs(grid, i + 1, j) + dfs(grid, i, j - 1) + dfs(grid, i, j + 1);
    }

public:
    int maxAreaOfIsland(vector<vector<int>>& grid) {
        int area = 0;
        for (int i = 0; i < grid.size(); i++) {
            for (int j = 0; j < grid[0].size(); j++) {
                if (grid[i][j] == 1) {
                    area = max(area, dfs(grid, i, j));
                }
            }
        }
        return area;
    }
};*/


//827. 最大人工岛
//我们先计算出所有岛屿的面积，在所有的格子上标记出岛屿的面积。
//然后搜索哪个海洋格子相邻的两个岛屿面积最大

//我们得能区分一个海洋格子相邻的两个格子是不是来自同一个岛屿。
//那么，我们不能在方格中标记岛屿的面积，而应该标记岛屿的索引（下标），
//另外用一个数组记录每个岛屿的面积

//可以看到，这道题实际上是对网格做了两遍 DFS：
//第一遍 DFS 遍历陆地格子，计算每个岛屿的面积并标记岛屿；
//第二遍 DFS 遍历海洋格子，观察每个海洋格子相邻的陆地格子。



//207. 课程表
// 我们使用一个队列来进行广度优先搜索。初始时，所有入度为0的节点都被放入队列中，
//它们就是可以作为拓扑排序最前面的节点，并且它们之间的相对顺序是无关紧要的。
//在广度优先搜索的每一步中，我们取出队首的节点u：我们将u 放入答案中；
//我们移除u 的所有出边，也就是将u 的所有相邻节点的入度减少1
//如果某个相邻节点v的入度变为0,那么我们就将v 放入队列中
//在广度优先搜索的过程结束后。如果答案中包含了这n 个节点，
//那么我们就找到了一种拓扑排序，否则说明图中存在环，也就不存在拓扑排序了

//广度优先搜索
/*
class Solution {
private:
    vector<vector<int>> edges;
    vector<int> indeg;

public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        edges.resize(numCourses);
        indeg.resize(numCourses);
        for (const auto& info : prerequisites) {
            edges[info[1]].push_back(info[0]);
            ++indeg[info[0]];
        }

        queue<int> q;
        for (int i = 0; i < numCourses; ++i) {
            if (indeg[i] == 0) {
                q.push(i);
            }
        }

        int visited = 0;
        while (!q.empty()) {
            ++visited;
            int u = q.front();
            q.pop();
            for (int v : edges[u]) {
                --indeg[v];
                if (indeg[v] == 0) {
                    q.push(v);
                }
            }
        }

        return visited == numCourses;
    }
};*/



//221. 最大正方形
//方法一：暴力法
//遍历矩阵中的每个元素，每次遇到1,则将该元素作为正方形的左上角；
//确定正方形的左上角后，根据左上角所在的行和列计算可能的最大正方形的边长（正方形的范围不能超出矩阵的行数和列数），
//在该边长范围内寻找只包含1的最大正方形；
//每次在下方新增一行以及在右方新增一列，判断新增的行和列是否满足所有元素都是1
/*
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        int maxSide = 0;
        int rows = matrix.size(), columns = matrix[0].size();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == '1') {
                    // 遇到一个 1 作为正方形的左上角
                    maxSide = max(maxSide, 1);
                    // 计算可能的最大正方形边长
                    int currentMaxSide = min(rows - i, columns - j);
                    for (int k = 1; k < currentMaxSide; k++) {
                        // 判断新增的一行一列是否均为 1
                        bool flag = true;
                        if (matrix[i + k][j + k] == '0') {
                            break;
                        }
                        for (int m = 0; m < k; m++) {
                            if (matrix[i + k][j + m] == '0' || matrix[i + m][j + k] == '0') {
                                flag = false;
                                break;
                            }
                        }
                        if (flag) {
                            maxSide = max(maxSide, k + 1);
                        }
                        else {
                            break;
                        }
                    }
                }
            }
        }
        int maxSquare = maxSide * maxSide;
        return maxSquare;
    }
};*/


//法二，动态规划法
//1）dp[i][j]:表示以（i，j）为右下角，且只包含1的正方形的边长的最大值
//2）递推公式:dp[i][j]=min(dp[i-1][j],dp[i-1][j-1],dp[i][j-1])+1
//3)初始化：dp[0][j]=1,dp[i][0]=1
//4）递推方向：从上到下。从左到右
//5）举例

/*
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            return 0;
        }
        int maxSide = 0;
        int rows = matrix.size(), columns = matrix[0].size();
        vector<vector<int>> dp(rows, vector<int>(columns));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                if (matrix[i][j] == '1') {
                    if (i == 0 || j == 0) {
                        dp[i][j] = 1;
                    }
                    else {
                        dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                    }
                    maxSide = max(maxSide, dp[i][j]);
                }
            }
        }
        int maxSquare = maxSide * maxSide;
        return maxSquare;
    }
};*/



//238. 除自身以外数组的乘积
//左右乘积列表
//时间复杂度:O(n)
//空间复杂度:O(n)
/*
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int length = nums.size();

        // L 和 R 分别表示左右两侧的乘积列表
        vector<int> L(length, 0), R(length, 0);

        vector<int> answer(length);

        // L[i] 为索引 i 左侧所有元素的乘积
        // 对于索引为 '0' 的元素，因为左侧没有元素，所以 L[0] = 1
        L[0] = 1;
        for (int i = 1; i < length; i++) {
            L[i] = nums[i - 1] * L[i - 1];
        }

        // R[i] 为索引 i 右侧所有元素的乘积
        // 对于索引为 'length-1' 的元素，因为右侧没有元素，所以 R[length-1] = 1
        R[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            R[i] = nums[i + 1] * R[i + 1];
        }

        // 对于索引 i，除 nums[i] 之外其余各元素的乘积就是左侧所有元素的乘积乘以右侧所有元素的乘积
        for (int i = 0; i < length; i++) {
            answer[i] = L[i] * R[i];
        }

        return answer;
    }
};*/


//空间优化
//时间复杂度:O(n)
//空间复杂度:O(1)
//由于输出数组不算在空间复杂度内，那么我们可以将 L 或 R 数组用输出数组来计算
//先把输出数组当作 L 数组来计算，然后再动态构造 R 数组得到结果。
/*
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int length = nums.size();
        vector<int> answer(length);

        // answer[i] 表示索引 i 左侧所有元素的乘积
        // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
        answer[0] = 1;
        for (int i = 1; i < length; i++) {
            answer[i] = nums[i - 1] * answer[i - 1];
        }

        // R 为右侧所有元素的乘积
        // 刚开始右边没有元素，所以 R = 1
        int R = 1;
        for (int i = length - 1; i >= 0; i--) {
            // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
            answer[i] = answer[i] * R;
            // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
            R *= nums[i];
        }
        return answer;
    }
};*/


//394. 字符串解码
//递归
/*
class Solution {
public:
    //k[dfs] 
    //u:当前处理到了什么位置
    string dfs(string& s, int& u) {
        string res;
        while (u < s.size()) {
            if (isdigit(s[u])) {
                int k = 0;
                while (u < s.size() && isdigit(s[u])) {
                    k = k * 10 + (s[u] - '0');
                    u++;
                }
                u++;//处理左括号
                string t = dfs(s, u);//取出[]里的字符
                while (k--) {
                    res += t;
                }
            }
            else if (isalpha(s[u])) {
                res += s[u++];
            }
            else if (s[u] == ']') {
                u++;
                return res;
            }
        }
        return res;
    }

    string decodeString(string& s) {
        int u = 0;
        return dfs(s, u);
    }
};*/

//处理左括号
/*
class Solution {
public:
    //k[dfs] 
    //u:当前处理到了什么位置
    string dfs(string& s, int& u) {
        string res;
        while (u < s.size()) {
            if (isdigit(s[u])) {
                int k = 0;
                while (u < s.size() && isdigit(s[u])) {
                    k = k * 10 + (s[u] - '0');
                    u++;
                }
                string t = dfs(s, u);//取出[]里的字符
                while (k--) {
                    res += t;
                }
            }
            else if (isalpha(s[u])) {
                res += s[u++];
            }
            else if (s[u] == ']') {
                u++;
                return res;
            }
            else if (s[u] == '[') {
                u++;
            }
        }
        return res;
    }

    string decodeString(string& s) {
        int u = 0;
        return dfs(s, u);
    }
};*/



/*
滑动窗口算法的大致逻辑如下：
1)left先不动，移动right指针,找到符合要求的子串
2）收缩left指针，找到最优解
3）重复1），2）
int left = 0, right = 0;

while (right < s.size()) {
    // 增大窗口
    window.add(s[right]);
    right++;

    while (window needs shrink) {
        // 缩小窗口
        window.remove(s[left]);
        left++;
    }
}*/




//* 滑动窗口算法框架 */
//*void slidingWindow(string s, string t) {
//    unordered_map<char, int> need, window;
//    for (char c : t) need[c]++;
//
//    int left = 0, right = 0;
//    int valid = 0;
//    while (right < s.size()) {
//        // c 是将移入窗口的字符
//        char c = s[right];
//        // 右移窗口
//        right++;
//        // 进行窗口内数据的一系列更新
//        ...
//
//            /*** debug 输出的位置 ***/
//            printf("window: [%d, %d)\n", left, right);
//        /********************/
//
//        // 判断左侧窗口是否要收缩
//        while (window needs shrink) {
//            // d 是将移出窗口的字符
//            char d = s[left];
//            // 左移窗口
//            left++;
//            // 进行窗口内数据的一系列更新
//            ...
//        }
//    }
//}

//思路：
// 现在开始套模板，只需要思考以下四个问题：

//1、当移动 right 扩大窗口，即加入字符时，应该更新哪些数据？
//2、什么条件下，窗口应该暂停扩大，开始移动 left 缩小窗口？
//3、当移动 left 缩小窗口，即移出字符时，应该更新哪些数据？
//4、我们要的结果应该在扩大窗口时还是缩小窗口时进行更新？


//如果一个字符进入窗口，应该增加 window 计数器；如果一个字符将移出窗口的时候，
//应该减少 window 计数器；当 valid 满足 need 时应该收缩窗口
//应该在收缩窗口的时候更新最终结果


//三、438. 找到字符串中所有字母异位词
//相当于，输入一个串 S，一个串 T，找到 S 中所有 T 的排列，返回它们的起始索引。
/*
vector<int> findAnagrams(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;//表示窗口中满足 need 条件的字符个数
    vector<int> res; // 记录结果
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c])
                valid++;
        }
        // 判断左侧窗口是否要收缩
        while (right - left >= t.size()) {
            // 当窗口符合条件时，把起始索引加入 res
            if (valid == need.size())
                res.push_back(left);
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    return res;
}*/


//一、76、最小覆盖子串
/*string minWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    // 记录最小覆盖子串的起始索引及长度
    int start = 0, len = INT_MAX;
    while (right < s.size()) {
        // c 是将移入窗口的字符
        char c = s[right];
        // 右移窗口
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c])
                valid++;
        }

        // 判断左侧窗口是否要收缩
        while (valid == need.size()) {
            // 在这里更新最小覆盖子串
            if (right - left < len) {
                start = left;
                len = right - left;
            }
            // d 是将移出窗口的字符
            char d = s[left];
            // 左移窗口
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    // 返回最小覆盖子串
    return len == INT_MAX ?
        "" : s.substr(start, len);
}*/



//二、567、字符串排列
// 判断 s 中是否存在 t 的排列
/*bool checkInclusion(string t, string s) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;

    int left = 0, right = 0;
    int valid = 0;
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c])
                valid++;
        }

        // 判断左侧窗口是否要收缩
        while (right - left >= t.size()) {
            // 在这里判断是否找到了合法的子串
            if (valid == need.size())
                return true;
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            if (need.count(d)) {
                if (window[d] == need[d])
                    valid--;
                window[d]--;
            }
        }
    }
    // 未找到符合条件的子串
    return false;
}*/


//四、3、最长无重复子串
/*int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> window;

    int left = 0, right = 0;
    int res = 0; // 记录结果
    while (right < s.size()) {
        char c = s[right];
        right++;
        // 进行窗口内数据的一系列更新
        window[c]++;
        // 判断左侧窗口是否要收缩
        while (window[c] > 1) {
            char d = s[left];
            left++;
            // 进行窗口内数据的一系列更新
            window[d]--;
        }
        // 在这里更新答案
        res = max(res, right - left);
    }
    return res;
}*/


//560. 和为 K 的子数组
//法一，暴力超时
//时间复杂度：O(n2)
//空间复杂度：O(1)
/*
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.size(); ++start) {
            int sum = 0;
            for (int end = start; end >= 0; --end) {
                sum += nums[end];
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
    }
};*/

//法二
//前缀和+哈希表，0<=j<=i
//建立哈希表 ，以和为键，出现次数为对应的值，记录pre[i] 出现的次数
//那么以i 结尾的答案mp[pre[i]−k] 即可在O(1) 时间内得到
/*class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1;
        int count = 0, pre = 0;
        for (auto& x : nums) {
            pre += x;
            if (mp.find(pre - k) != mp.end()) {
                count += mp[pre - k];
            }
            mp[pre]++;
        }
        return count;
    }
};*/


//581. 最短无序连续子数组
//排序，与原数组比较，记下相同的前缀和后缀长度，总的长度一剪就行
//时间复杂度：O(nlogn)
//空间复杂度：O(n)
/*
class Solution {
public:
    int findUnsortedSubarray(vector<int>& nums) {
        if (is_sorted(nums.begin(), nums.end())) {
            return 0;
        }
        vector<int> numsSorted(nums);
        sort(numsSorted.begin(), numsSorted.end());
        int left = 0;
        while (nums[left] == numsSorted[left]) {
            left++;
        }
        int right = nums.size() - 1;
        while (nums[right] == numsSorted[right]) {
            right--;
        }
        return right - left + 1;
    }
};*/


//621. 任务调度器
//两个 相同种类 的任务之间必须有长度为整数 n 的冷却时间，
//因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

//桶思想
//建立大小为 n + 1 的桶子，个数为任务数量最多的那个任务
//记录最大任务数量 N，看一下任务数量并列最多的任务有多少个，即最后一个桶子的任务数 X，
//计算 NUM1 = (N - 1) * (n + 1) + x
//NUM2 = tasks.size()输出其中较大值即可
//因为存在空闲时间时肯定是 NUM1 大，不存在空闲时间时肯定是 NUM2 >= NUM1

//总结起来：总排队时间 = (桶个数 - 1) * (n + 1) + 最后一桶的任务数

/*
int leastInterval(vector<char>& tasks, int n) {
    int len = tasks.size();
    vector<int> vec(26);
    for (char c : tasks) ++vec[c - 'A'];
    sort(vec.begin(), vec.end(), [](int& x, int& y) {return x > y; });
    int cnt = 1;
    while (cnt < vec.size() && vec[cnt] == vec[0]) cnt++;
    return max(len, cnt + (n + 1) * (vec[0] - 1));
}*/