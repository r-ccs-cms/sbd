/// This is a part of qscd
/**
@file det_vector.h
@brief flat-packed fixed-row-width replacement for std::vector<std::vector<ElemT>>
 */

#ifndef SBD_FRAMEWORK_DET_VECTOR_H
#define SBD_FRAMEWORK_DET_VECTOR_H

#include <algorithm>
#include <cstring>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace sbd {

enum class det_kind { full, half };

// Forward declaration so detail function declarations below can name det_vector.
template<typename ElemT, det_kind Kind>
class det_vector;

namespace detail {
template<size_t Mask, typename ElemT, det_kind Kind>
void idx_sort_det(std::vector<int>&, const det_vector<ElemT, Kind>&, int, int, int);
template<size_t Mask, bool FillCounts, typename ElemT, det_kind Kind>
void unique_from_sorted_impl(const det_vector<ElemT, Kind>&, det_vector<ElemT, Kind>&, std::vector<int>&);
template<size_t Mask, bool FillCounts, typename ElemT, det_kind Kind>
void unique_from_unsorted_impl(const det_vector<ElemT, Kind>&, det_vector<ElemT, Kind>&, std::vector<int>&);
} // namespace detail

// Flat-packed container where every row has the same fixed width _elem_size.
// Storage: std::vector<ElemT> _data of length size * _elem_size (tightly packed).
//   _data[i*_elem_size .. i*_elem_size+_elem_size-1] = row i data
// Only intended for ElemT = size_t or uint32_t.
// _elem_size is a shared static per specialisation: det_vector<ElemT, Kind>
// instances all carry the same row width.  It is set once (from 0) and never changed.
// The Kind tag generates distinct specialisations with independent _elem_size values,
// allowing full-det and half-det containers to coexist without aliasing.
template<typename ElemT, det_kind Kind = det_kind::full>
class det_vector {
    static_assert(std::is_trivially_copyable_v<ElemT>,
                  "det_vector<ElemT> requires a trivially copyable element type");
public:

    // Non-owning view of a single row obtained via reinterpret_cast from the
    // backing store.  _data[k] is data element k — out-of-bounds for the declared
    // array but backed by the flat allocation (C struct-hack pattern).
    // Constructors are deleted so standalone row objects cannot be created.
    struct row {
        ElemT _data[1];

        row()              = delete;
        row(const row&)    = delete;

        size_t size() const noexcept { return det_vector::_elem_size; }

        // No-op: row width is fixed by det_vector::_elem_size.  Exists so that
        // generic 2D-container code (e.g. for (auto& r : c) r.resize(n)) compiles.
        void resize(size_t n) {
            if (n != size())
                throw std::length_error("det_vector::row: resize to wrong size");
        }

        ElemT& operator[](size_t k) noexcept       { return _data[k]; }
        const ElemT& operator[](size_t k) const noexcept { return _data[k]; }

        ElemT* data() noexcept             { return _data; }
        const ElemT* data() const noexcept { return _data; }

        ElemT* begin() noexcept             { return _data; }
        ElemT* end()   noexcept             { return _data + size(); }
        const ElemT* begin() const noexcept { return _data; }
        const ElemT* end()   const noexcept { return _data + size(); }

        operator std::vector<ElemT>() const {
            return std::vector<ElemT>(_data, _data + size());
        }

        // In-place copy from vector; row size must match.
        row& operator=(const std::vector<ElemT>& v) {
            if (v.size() != size())
                throw std::length_error("det_vector::row: vector size mismatch");
            std::memcpy(_data, v.data(), size() * sizeof(ElemT));
            return *this;
        }

        // In-place copy from another row; memcpy of full row data.
        row& operator=(const row& other) {
            if (this == &other) return *this;
            std::memcpy(_data, other._data, size() * sizeof(ElemT));
            return *this;
        }

        bool operator==(const std::vector<ElemT>& v) const noexcept {
            if (v.size() != size()) return false;
            return std::memcmp(_data, v.data(), size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const std::vector<ElemT>& v) const noexcept {
            return !(*this == v);
        }
        bool operator==(const row& other) const noexcept {
            return std::memcmp(_data, other._data, size() * sizeof(ElemT)) == 0;
        }
        bool operator!=(const row& other) const noexcept {
            return !(*this == other);
        }
        // From-back (little-endian multi-word) order: last element is most significant.
        // Consistent with less_from_back() in bit_manipulation.h and with sbd::sort().
        // std::lower_bound(da.begin(), da.end(), std::vector<ElemT> v) works directly
        // after sbd::sort() because lower_bound calls *it < v → this operator.
        bool operator<(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = n; k > 0; k--) {
                if (_data[k-1] < v[k-1]) return true;
                if (_data[k-1] > v[k-1]) return false;
            }
            return false;
        }
        bool operator<(const row& other) const noexcept {
            size_t n = size();
            for (size_t k = n; k > 0; k--) {
                if (_data[k-1] < other._data[k-1]) return true;
                if (_data[k-1] > other._data[k-1]) return false;
            }
            return false;
        }
        bool operator>(const std::vector<ElemT>& v) const noexcept {
            size_t n = size();
            for (size_t k = n; k > 0; k--) {
                if (_data[k-1] > v[k-1]) return true;
                if (_data[k-1] < v[k-1]) return false;
            }
            return false;
        }
        bool operator<=(const std::vector<ElemT>& v) const noexcept { return !(*this > v); }
        bool operator>=(const std::vector<ElemT>& v) const noexcept { return !(*this < v); }
        bool operator>(const row& other) const noexcept { return other < *this; }
        bool operator<=(const row& other) const noexcept { return !(other < *this); }
        bool operator>=(const row& other) const noexcept { return !(*this < other); }
        // Reversed-argument overloads so vector op row works via ADL.
        friend bool operator==(const std::vector<ElemT>& v, const row& r) noexcept { return r == v; }
        friend bool operator!=(const std::vector<ElemT>& v, const row& r) noexcept { return r != v; }
        friend bool operator< (const std::vector<ElemT>& v, const row& r) noexcept { return r >  v; }
        friend bool operator> (const std::vector<ElemT>& v, const row& r) noexcept { return r <  v; }
        friend bool operator<=(const std::vector<ElemT>& v, const row& r) noexcept { return r >= v; }
        friend bool operator>=(const std::vector<ElemT>& v, const row& r) noexcept { return r <= v; }

        friend void swap(row& a, row& b) noexcept {
            for (size_t k = 0, n = a.size(); k < n; k++) std::swap(a[k], b[k]);
        }
    };

    using value_type = row;

    // Random-access iterator. _ptr points to _data[0] of the current row.
    // Stride is taken directly from det_vector::_elem_size (shared static) so
    // the iterator carries only one pointer word.
    struct iterator {
        using value_type        = row;
        using reference         = row&;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::random_access_iterator_tag;
        using pointer           = row*;

        ElemT*  _ptr;

        row& operator*()  const noexcept {
            return *reinterpret_cast<row*>(_ptr);
        }
        row& operator[](difference_type n) const noexcept {
            return *reinterpret_cast<row*>(
                _ptr + n * static_cast<difference_type>(_elem_size));
        }
        row* operator->() const noexcept {
            return reinterpret_cast<row*>(_ptr);
        }

        iterator& operator++() noexcept { _ptr += _elem_size; return *this; }
        iterator  operator++(int) noexcept { auto t = *this; _ptr += _elem_size; return t; }
        iterator& operator--() noexcept { _ptr -= _elem_size; return *this; }
        iterator  operator--(int) noexcept { auto t = *this; _ptr -= _elem_size; return t; }
        iterator& operator+=(difference_type n) noexcept {
            _ptr += n * static_cast<difference_type>(_elem_size); return *this;
        }
        iterator& operator-=(difference_type n) noexcept {
            _ptr -= n * static_cast<difference_type>(_elem_size); return *this;
        }
        iterator operator+(difference_type n) const noexcept {
            return {_ptr + n * static_cast<difference_type>(_elem_size)};
        }
        iterator operator-(difference_type n) const noexcept {
            return {_ptr - n * static_cast<difference_type>(_elem_size)};
        }
        difference_type operator-(const iterator& o) const noexcept {
            auto diff = _ptr - o._ptr;
            return diff == 0 ? 0 : diff / static_cast<difference_type>(_elem_size);
        }
        friend iterator operator+(difference_type n, const iterator& it) noexcept {
            return it + n;
        }

        bool operator==(const iterator& o) const noexcept { return _ptr == o._ptr; }
        bool operator!=(const iterator& o) const noexcept { return _ptr != o._ptr; }
        bool operator< (const iterator& o) const noexcept { return _ptr <  o._ptr; }
        bool operator> (const iterator& o) const noexcept { return _ptr >  o._ptr; }
        bool operator<=(const iterator& o) const noexcept { return _ptr <= o._ptr; }
        bool operator>=(const iterator& o) const noexcept { return _ptr >= o._ptr; }
    };

    // --- construction ---

    det_vector() = default;
    explicit det_vector(size_t n) { resize(n); }

    // Construct n rows, each a copy of v. Sets _elem_size from v.size().
    det_vector(size_t n, const std::vector<ElemT>& v)
        : _size(n)
    {
        _set_elem_size(v.size());
        _data.resize(n * stride());
        for (size_t i = 0; i < n; i++) _init_row(i, v.data());
    }

    // Specialization for det_vector::iterator: rows are already guaranteed to be
    // the right width so a single memcpy of the flat backing store suffices.
    det_vector(iterator first, iterator last) {
        if (first == last) return;
        size_t n = static_cast<size_t>(last - first);
        _data.resize(n * _elem_size);
        _size = n;
        std::memcpy(_data.data(), first._ptr, n * _elem_size * sizeof(ElemT));
    }

    template<typename InputIt>
    det_vector(InputIt first, InputIt last) {
        if (first == last) return;
        size_t n = static_cast<size_t>(std::distance(first, last));
        _set_elem_size(first->size());
        _data.resize(n * stride());
        _size = n;
        size_t i = 0;
        for (auto it = first; it != last; ++it, ++i) {
            if (it->size() != _elem_size)
                throw std::length_error("det_vector: row size mismatch in range constructor");
            std::copy(it->begin(), it->end(), _data.data() + i * stride());
        }
    }

    det_vector(const det_vector&) = default;
    det_vector& operator=(const det_vector&) = default;

    det_vector(det_vector&& other) noexcept
        : _data(std::move(other._data)),
          _size(other._size)
    {
        other._data.clear();
        other._size = 0;
    }

    det_vector& operator=(det_vector&& other) noexcept {
        if (this != &other) {
            _data  = std::move(other._data);
            _size  = other._size;
            other._data.clear();
            other._size = 0;
        }
        return *this;
    }

    // --- observers ---

    size_t size()      const noexcept { return _size; }
    size_t elem_size() const noexcept { return _elem_size; }
    bool   empty()     const noexcept { return _size == 0; }

    // Direct access to the flat backing buffer (_size * _elem_size elements).
    std::vector<ElemT>&       flat()        noexcept { return _data; }
    const std::vector<ElemT>& cflat() const noexcept { return _data; }

    // --- element access ---

    row& operator[](size_t i) noexcept {
        return *reinterpret_cast<row*>(_data.data() + i * stride());
    }
    const row& operator[](size_t i) const noexcept {
        return *reinterpret_cast<const row*>(_data.data() + i * stride());
    }

    // --- iterators ---

    iterator begin() noexcept {
        return iterator{_data.data()};
    }
    iterator end() noexcept {
        return iterator{_data.data() + _size * stride()};
    }
    // const begin/end return the same iterator type; row& from a const det_vector
    // is the pragmatic research-code tradeoff (no separate const_iterator).
    iterator begin() const {
        return iterator{const_cast<ElemT*>(_data.data())};
    }
    iterator end() const {
        return iterator{const_cast<ElemT*>(_data.data()) + _size * stride()};
    }

    // --- modifiers ---

    // Set _size = 0 and release _data storage.
    void clear() noexcept { _data.clear(); _size = 0; }

    // Reserve capacity for at least n rows without changing _size or initialising rows.
    void reserve(size_t n) { _data.reserve(n * stride()); }

    size_t capacity() const noexcept {
        return (_elem_size > 0) ? (_data.capacity() / stride()) : 0;
    }

    // Resize to n rows. _data.size() is always exactly n * _elem_size.
    // New rows (if any) are left uninitialised. Requires _elem_size already set when n > 0.
    void resize(size_t n) {
        if (n > 0 && _elem_size == 0)
            throw std::length_error("det_vector: elem_size not set");
        _data.resize(n * stride());
        _size = n;
    }

    void resize(size_t n, const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        size_t old_size = _size;
        _data.resize(n * stride());
        _size = n;
        for (size_t i = old_size; i < n; ++i) _init_row(i, v.data());
    }

    void swap(det_vector& other) noexcept {
        _data.swap(other._data);
        std::swap(_size, other._size);
    }

    // Replace contents with m rows each initialised to v. Sets _elem_size from v.size().
    void assign(size_t m, const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        _size = m;
        _data.resize(m * stride());
        for (size_t i = 0; i < m; i++) _init_row(i, v.data());
    }

    // Specialization for det_vector::iterator: single memcpy, no per-row check needed.
    void assign(iterator first, iterator last) {
        size_t n = static_cast<size_t>(last - first);
        _data.resize(n * _elem_size);
        _size = n;
        if (n > 0) std::memcpy(_data.data(), first._ptr, n * _elem_size * sizeof(ElemT));
    }

    // Replace contents from [first, last). Sets _elem_size from first row, checks all rows.
    template<typename InputIt>
    void assign(InputIt first, InputIt last) {
        if (first == last) { _data.clear(); _size = 0; return; }
        size_t n = static_cast<size_t>(std::distance(first, last));
        _set_elem_size(first->size());
        _data.resize(n * stride());
        _size = n;
        size_t i = 0;
        for (auto it = first; it != last; ++it, ++i) {
            if (it->size() != _elem_size)
                throw std::length_error("det_vector: row size mismatch in assign");
            std::copy(it->begin(), it->end(), _data.data() + i * stride());
        }
    }

    void push_back(const std::vector<ElemT>& v) {
        _set_elem_size(v.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, v.data());
        ++_size;
    }

    void push_back(const row& r) {
        _set_elem_size(r.size());
        _data.resize((_size + 1) * stride());
        _init_row(_size, r._data);
        ++_size;
    }

    void emplace_back(const std::vector<ElemT>& v) { push_back(v); }
    void emplace_back(const row& r) { push_back(r); }

    // Append a row from an iterator range [first, last).
    template<typename InputIt>
    void emplace_back(InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        _set_elem_size(n);
        _data.resize((_size + 1) * stride());
        std::copy(first, last, _data.data() + _size * stride());
        ++_size;
    }

    // Insert rows from [first, last) before pos. Each source row must have
    // size() == _elem_size. Works for both det_vector::iterator and
    // std::vector<std::vector<ElemT>>::iterator source ranges.
    template<typename InputIt>
    iterator insert(iterator pos, InputIt first, InputIt last) {
        size_t n = static_cast<size_t>(std::distance(first, last));
        if (n == 0) return pos;
        size_t pos_idx = static_cast<size_t>(pos - begin());
        size_t old_size = _size;
        _data.resize((_size + n) * stride());
        _size += n;
        ElemT* base = _data.data();
        if (pos_idx < old_size) {
            std::memmove(base + (pos_idx + n) * stride(),
                         base + pos_idx * stride(),
                         (old_size - pos_idx) * stride() * sizeof(ElemT));
        }
        ElemT* dst = base + pos_idx * stride();
        for (auto it = first; it != last; ++it, dst += stride()) {
            const auto& src = *it;
            std::copy(src.begin(), src.end(), dst);
        }
        return iterator{base + pos_idx * stride()};
    }

    iterator erase(iterator first, iterator last) {
        if (first == last) return last;
        size_t pos  = static_cast<size_t>(first - begin());
        size_t n    = static_cast<size_t>(last  - first);
        size_t tail = _size - (pos + n);
        if (tail > 0)
            std::memmove(_data.data() + pos * stride(),
                         _data.data() + (pos + n) * stride(),
                         tail * stride() * sizeof(ElemT));
        _size -= n;
        _data.resize(_size * stride());
        return iterator{_data.data() + pos * stride()};
    }

    iterator erase(iterator pos) { return erase(pos, pos + 1); }

    // In-place unique / sort — member functions.
    // Called by the free-function overloads when &in == &out, and directly by callers
    // that already have a mutable det_vector and want in-place semantics.

    template<size_t Mask = ~size_t(0)>
    void unique_from_sorted() {
        std::vector<int> dummy;
        detail::unique_from_sorted_impl<Mask, false>(
            static_cast<const det_vector&>(*this), *this, dummy);
    }

    template<size_t Mask = ~size_t(0)>
    void unique_with_counts_from_sorted(std::vector<int>& counts) {
        detail::unique_from_sorted_impl<Mask, true>(
            static_cast<const det_vector&>(*this), *this, counts);
    }

    template<size_t Mask = ~size_t(0)>
    void unique_from_unsorted() {
        int n = (int)_size, row_len = (int)_elem_size;
        if (n == 0) return;
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        if (n > 1) detail::idx_sort_det<Mask>(idx, *this, 0, n, row_len - 1);
        // Collect unique source positions in idx[0..m-1] in one pass.
        // idx[i-1] is always the original sorted position at i-1 because writes
        // only go to idx[0..m-1] where m < i at the time of each write.
        auto eq_prev = [&](int i) {
            for (int k = row_len; k > 0; k--)
                if (((*this)[idx[i]][k-1] & Mask) != ((*this)[idx[i-1]][k-1] & Mask))
                    return false;
            return true;
        };
        int m = 1;
        for (int i = 1; i < n; i++)
            if (!eq_prev(i)) idx[m++] = idx[i];
        // Save m unique rows, shrink _data in place (reuses existing capacity),
        // then copy back from scratch.
        std::vector<ElemT> buf(m * _elem_size);
        for (int j = 0; j < m; j++)
            std::memcpy(buf.data() + j * _elem_size,
                        _data.data() + idx[j] * _elem_size,
                        _elem_size * sizeof(ElemT));
        _data.resize(m * _elem_size);
        _size = m;
        std::memcpy(_data.data(), buf.data(), m * _elem_size * sizeof(ElemT));
    }

    template<size_t Mask = ~size_t(0)>
    void unique_with_counts_from_unsorted(std::vector<int>& counts) {
        int n = (int)_size, row_len = (int)_elem_size;
        if (n == 0) { counts.clear(); return; }
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        if (n > 1) detail::idx_sort_det<Mask>(idx, *this, 0, n, row_len - 1);
        auto eq_prev = [&](int i) {
            for (int k = row_len; k > 0; k--)
                if (((*this)[idx[i]][k-1] & Mask) != ((*this)[idx[i-1]][k-1] & Mask))
                    return false;
            return true;
        };
        // Pass 1: count unique (read-only on idx) so counts can be sized exactly.
        int m = 1;
        for (int i = 1; i < n; i++) if (!eq_prev(i)) m++;
        counts.resize(m);
        // Pass 2: collect unique source positions in idx[0..m-1] and fill counts.
        // idx[i-1] is always the original sorted position at i-1 because writes
        // only go to idx[0..w-1] where w < i at the time of each write.
        int w = 1, cw = 0, cur = 1;
        for (int i = 1; i < n; i++) {
            if (!eq_prev(i)) { counts[cw++] = cur; idx[w++] = idx[i]; cur = 1; }
            else cur++;
        }
        counts[cw] = cur;
        std::vector<ElemT> buf(m * _elem_size);
        for (int j = 0; j < m; j++)
            std::memcpy(buf.data() + j * _elem_size,
                        _data.data() + idx[j] * _elem_size,
                        _elem_size * sizeof(ElemT));
        _data.resize(m * _elem_size);
        _size = m;
        std::memcpy(_data.data(), buf.data(), m * _elem_size * sizeof(ElemT));
    }

    // In-place sort in from-back/Mask order.
    // Builds a sorted index permutation, then applies it via cycle following
    // using a single row of scratch — no full-sized temporary allocation.
    template<size_t Mask = ~size_t(0)>
    void sort() {
        int n       = (int)_size;
        int row_len = (int)_elem_size;
        if (n <= 1) return;
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        detail::idx_sort_det<Mask>(idx, *this, 0, n, row_len - 1);
        // Apply permutation in-place: follow each cycle with one row of scratch.
        // Negate idx[j] to mark positions already placed in their final location.
        std::vector<ElemT> buf(_elem_size);
        ElemT* base = _data.data();
        size_t stride = _elem_size;
        for (int i = 0; i < n; i++) {
            if (idx[i] < 0 || idx[i] == i) continue;
            std::memcpy(buf.data(), base + i * stride, stride * sizeof(ElemT));
            int j = i;
            while (idx[j] != i) {
                int k = idx[j];
                std::memcpy(base + j * stride, base + k * stride, stride * sizeof(ElemT));
                idx[j] = ~idx[j];
                j = k;
            }
            std::memcpy(base + j * stride, buf.data(), stride * sizeof(ElemT));
            idx[j] = ~idx[j];
        }
    }

    // Call once at startup with the known row width before any container is used.
    // No-op on repeated calls with the same value; asserts if called with a different value.
    static void init_elem_size(size_t n) { _set_elem_size(n); }

private:
    std::vector<ElemT> _data;
    size_t _size = 0;

    inline static size_t _elem_size = 0;

    // Set _elem_size to n. No-op if already n; throws if non-zero and different.
    static void _set_elem_size(size_t n) {
        if (_elem_size == n) return;
        if (_elem_size != 0)
            throw std::length_error("det_vector: elem_size mismatch");
        _elem_size = n;
    }

    size_t stride() const noexcept { return _elem_size; }

    // Write data for row i from src (length _elem_size).
    void _init_row(size_t i, const ElemT* src) noexcept {
        std::memcpy(_data.data() + i * stride(), src, _elem_size * sizeof(ElemT));
    }
};

// ---- sort / unique free functions ----
//
// sort<Mask>(in, out)                              — sort into out; delegates to member if aliased.
// da.sort<Mask>()                                  — sort in-place (member).
// unique_from_sorted<Mask>(in, out)                — dedup sorted input, no counts; delegates if aliased.
// unique_with_counts_from_sorted<Mask>(in,out,cnt) — same, fills counts; delegates if aliased.
// unique_from_unsorted<Mask>(in, out)              — sort-then-dedup, no counts; delegates if aliased.
// unique_with_counts_from_unsorted<Mask>(in,out,c) — same, fills counts; delegates if aliased.
// da.unique_from_sorted<Mask>()                    — dedup sorted in-place (member).
// da.unique_with_counts_from_sorted<Mask>(counts)  — same, fills counts (member).
// da.unique_from_unsorted<Mask>()                  — sort-then-dedup in-place (member).
// da.unique_with_counts_from_unsorted<Mask>(counts)— same, fills counts (member).
//
// Mask: bitmask applied per ElemT word before comparison (default ~0 = no mask).
//   Mask = 0x5555555555555555ULL restricts to even-bit (alpha) positions.
//   Specialises at compile time; Mask=~0 eliminates the AND entirely.
//
// counts: output — counts[j] = occurrences of out[j] in in.
//   Resized to exactly the number of unique rows; existing storage reused if
//   capacity allows.
//
// Aliasing: all free functions detect &in == &out and delegate to the corresponding
//   member function rather than silently producing wrong output.
//   Sorted in-place member: two-pass, w<=i invariant, no temporary.
//   Unsorted in-place member: temporary of exactly m rows (arbitrary permutation).

namespace detail {

// Sort idx[lo..hi) so that (a[idx[i]][elem] & Mask) is in ascending order,
// recursing on equal-valued runs through decreasing elem (from-back ordering).
// Mask is a template parameter: Mask=~0 is compiled as a no-op by the optimiser.
template<size_t Mask, typename ElemT, det_kind Kind>
void idx_sort_det(std::vector<int>& idx,
                   const det_vector<ElemT, Kind>& a,
                   int lo, int hi, int elem) {
    if (hi - lo <= 1 || elem < 0) return;
    std::sort(idx.begin() + lo, idx.begin() + hi,
              [&a, elem](int x, int y) noexcept {
                  return (a[x][elem] & Mask) < (a[y][elem] & Mask);
              });
    if (elem == 0) return;
    int run_lo = lo;
    for (int i = lo + 1; i <= hi; i++) {
        if (i == hi ||
            (a[idx[i  ]][elem] & Mask) !=
            (a[idx[run_lo]][elem] & Mask)) {
            if (i - run_lo > 1)
                idx_sort_det<Mask>(idx, a, run_lo, i, elem - 1);
            run_lo = i;
        }
    }
}

// Two-pass dedup of pre-sorted input.  Pass 1 counts unique rows; pass 2 writes
// them into out sized to exactly m.  For aliased (&in==&out) the write-position
// invariant (w <= i for sorted input) makes in-place writes safe; out is resized
// to trim the tail only after all writes complete.
template<size_t Mask, bool FillCounts, typename ElemT, det_kind Kind>
void unique_from_sorted_impl(const det_vector<ElemT, Kind>& in,
                              det_vector<ElemT, Kind>& out,
                              std::vector<int>& counts) {
    int n       = (int)in.size();
    int row_len = (int)in.elem_size();
    if (n == 0) {
        out.clear();
        if constexpr (FillCounts) counts.clear();
        return;
    }

    auto eq_prev = [&](int i) {
        for (int k = row_len; k > 0; k--)
            if ((in[i][k-1] & Mask) != (in[i-1][k-1] & Mask)) return false;
        return true;
    };

    // Pass 1: count unique elements.
    int m = 1;
    for (int i = 1; i < n; i++)
        if (!eq_prev(i)) m++;

    if constexpr (FillCounts) counts.resize(m);

    // Pass 2: write unique rows.
    // Non-aliased: resize out first then write.
    // Aliased: write in-place (w <= i guaranteed for sorted input), resize last.
    int w = 0, cur_count = 1;
    if (&in != &out) {
        out.resize(m, row_len);
        out[0] = in[0];
    }
    for (int i = 1; i < n; i++) {
        if (!eq_prev(i)) {
            if constexpr (FillCounts) counts[w] = cur_count;
            w++;
            out[w] = in[i];
            cur_count = 1;
        } else cur_count++;
    }
    if constexpr (FillCounts) counts[w] = cur_count;
    if (&in == &out) out.resize(m, row_len);
}

// Two-pass dedup of unsorted input.  Builds a sorted index permutation, then
// counts unique rows, resizes out to exactly m, and writes.  Aliased case
// uses a temporary of size m to avoid source/dest overlap under the permutation.
template<size_t Mask, bool FillCounts, typename ElemT, det_kind Kind>
void unique_from_unsorted_impl(const det_vector<ElemT, Kind>& in,
                                det_vector<ElemT, Kind>& out,
                                std::vector<int>& counts) {
    int n       = (int)in.size();
    int row_len = (int)in.elem_size();
    if (n == 0) {
        out.clear();
        if constexpr (FillCounts) counts.clear();
        return;
    }

    // Build sorted index permutation.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (n > 1) idx_sort_det<Mask>(idx, in, 0, n, row_len - 1);

    auto eq_prev = [&](int i) {
        for (int k = row_len; k > 0; k--)
            if ((in[idx[i]][k-1] & Mask) != (in[idx[i-1]][k-1] & Mask)) return false;
        return true;
    };

    // Pass 1: count unique elements.
    int m = 1;
    for (int i = 1; i < n; i++)
        if (!eq_prev(i)) m++;

    if constexpr (FillCounts) counts.resize(m);

    // Pass 2: write unique rows via sorted permutation.
    auto write_to = [&](det_vector<ElemT, Kind>& dst) {
        dst.resize(m, row_len);
        int w = 0, cur_count = 1;
        dst[0] = in[idx[0]];
        for (int i = 1; i < n; i++) {
            if (!eq_prev(i)) {
                if constexpr (FillCounts) counts[w] = cur_count;
                w++;
                dst[w] = in[idx[i]];
                cur_count = 1;
            } else cur_count++;
        }
        if constexpr (FillCounts) counts[w] = cur_count;
    };

    if (&in == &out) {
        det_vector<ElemT, Kind> tmp(0, (size_t)row_len);
        write_to(tmp);
        out = std::move(tmp);
    } else {
        write_to(out);
    }
}

} // namespace detail

// sort: aliased (&in == &out) delegates to the in-place member function.
template<size_t Mask = ~size_t(0), typename ElemT, det_kind Kind>
void sort(const det_vector<ElemT, Kind>& in, det_vector<ElemT, Kind>& out) {
    if (&in == &out) { out.template sort<Mask>(); return; }
    int n       = (int)in.size();
    int row_len = (int)in.elem_size();
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    if (n > 1)
        detail::idx_sort_det<Mask>(idx, in, 0, n, row_len - 1);
    out.resize(n, row_len);
    for (int i = 0; i < n; i++) out[i] = in[idx[i]];
}

// unique_from_sorted: precondition — in is sorted in from-back/Mask order.
// Aliased (&in == &out) delegates to the in-place member function.
template<size_t Mask = ~size_t(0), typename ElemT, det_kind Kind>
void unique_from_sorted(const det_vector<ElemT, Kind>& in, det_vector<ElemT, Kind>& out) {
    if (&in == &out) { out.template unique_from_sorted<Mask>(); return; }
    std::vector<int> dummy;
    detail::unique_from_sorted_impl<Mask, false>(in, out, dummy);
}

// unique_with_counts_from_sorted: counts[j] = occurrences of out[j] in in.
// Aliased (&in == &out) delegates to the in-place member function.
template<size_t Mask = ~size_t(0), typename ElemT, det_kind Kind>
void unique_with_counts_from_sorted(const det_vector<ElemT, Kind>& in,
                                     det_vector<ElemT, Kind>& out,
                                     std::vector<int>& counts) {
    if (&in == &out) { out.template unique_with_counts_from_sorted<Mask>(counts); return; }
    detail::unique_from_sorted_impl<Mask, true>(in, out, counts);
}

// unique_from_unsorted: no precondition on sort order; sorts internally.
// Aliased (&in == &out) delegates to the in-place member function.
template<size_t Mask = ~size_t(0), typename ElemT, det_kind Kind>
void unique_from_unsorted(const det_vector<ElemT, Kind>& in, det_vector<ElemT, Kind>& out) {
    if (&in == &out) { out.template unique_from_unsorted<Mask>(); return; }
    std::vector<int> dummy;
    detail::unique_from_unsorted_impl<Mask, false>(in, out, dummy);
}

// unique_with_counts_from_unsorted: counts[j] = occurrences of out[j] in in.
// Aliased (&in == &out) delegates to the in-place member function.
template<size_t Mask = ~size_t(0), typename ElemT, det_kind Kind>
void unique_with_counts_from_unsorted(const det_vector<ElemT, Kind>& in,
                                       det_vector<ElemT, Kind>& out,
                                       std::vector<int>& counts) {
    if (&in == &out) { out.template unique_with_counts_from_unsorted<Mask>(counts); return; }
    detail::unique_from_unsorted_impl<Mask, true>(in, out, counts);
}

} // namespace sbd

#endif // SBD_FRAMEWORK_DET_VECTOR_H
