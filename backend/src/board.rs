#![allow(dead_code)]
use super::utils::*;
use std::cmp::Ordering;
use std::cmp::*;

#[derive(Debug)]
struct Line {
    count: u32,
    space_count: u32,
    open_count: u32,
}

impl Line {
    pub fn new(count: u32, space_count: u32, open_count: u32) -> Self {
        Self {
            count: count,
            space_count: space_count,
            open_count: open_count,
        }
    }

    pub fn is_winner_step(&self) -> bool {
        self.count >= 5 && self.space_count == 0
    }

    pub fn is_non_refutable(&self) -> bool {
        self.count == 4 && self.space_count == 0 && self.open_count == 2
    }

    pub fn must_be_blocked(&self) -> bool {
        match (self.count, self.space_count, self.open_count) {
            (3, 0, 2) => true,
            (3, 1, 2) => true,
            (4, 1, _) => true,
            (4, 0, v) if v > 0 => true,
            (_, _, _) => false,
        }
    }

    pub fn score(&self) -> u32 {
        match (self.count, self.space_count, self.open_count) {
            (v, 0, _) if v >= 5 => 11000,
            (v, 1, _) if v >= 5 => 2000,
            (4, 0, 2) => 10000,
            (3, 0, 2) => 50,
            (4, 0, 1) => 50,
            (4, 1, 1) => 30,
            (4, 1, 2) => 30,
            (3, 0, 1) => 30,
            (3, 1, 2) => 10,
            (3, 1, 1) => 25,
            (2, 0, 2) => 25,
            (_, _, _) => 0,
        }
    }

    fn single_score(&self) -> u32 {
        self.count * 2 + ((self.open_count as f32) * 1.5) as u32 - self.space_count
    }

    pub fn cmp(&self, other: &Line) -> Ordering {
        self.single_score().cmp(&other.single_score())
    }
}

impl PartialEq for Line {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count
            && self.space_count == other.space_count
            && self.open_count == other.open_count
    }
}

#[derive(Debug, Clone)]
pub struct Board {
    pub width: usize,
    pub height: usize,
    cells: Vec<Vec<u8>>,
    at_x: i32,
    at_y: i32,
}

impl From<String> for Board {
    fn from(input: String) -> Self {
        let len = input.len();
        let width = (len as f64).sqrt() as usize;
        let height = width;
        if width * height != len {
            panic!("Invalid input string size");
        }
        Board::new(input, width, height)
    }
}

impl Board {
    pub fn new(input: String, width: usize, height: usize) -> Self {
        let rows: Vec<u8> = input
            .chars()
            .map(|e| {
                if e == '.' {
                    '0'
                } else if e == 'o' {
                    '1'
                } else if e == '+' {
                    '2'
                } else {
                    e
                }
            })
            .filter(|&e| e == '0' || e == '1' || e == '2')
            .map(|e| e as u8 - '0' as u8)
            .collect();
        if rows.len() != width * height {
            panic!(
                "Invalid board size with {}*{} <> {}",
                width,
                height,
                rows.len()
            );
        }
        if width < 5 && height < 5 {
            panic!("Width or height must larger than")
        }
        Self {
            width: width,
            height: height,
            cells: rows.chunks(width).map(|x| x.to_vec()).collect(),
            at_x: -1,
            at_y: -1,
        }
    }
    pub fn any_winner(&self) -> Option<u8> {
        for i in 0..self.height {
            for j in 0..self.width {
                for p in 1..3 {
                    if self.get(i as i32, j as i32) == Some(p) {
                        for k in 0..4 {
                            let d = cfg::DIRS[k];
                            let line = self.connect_direction(p, i, j, d[0], d[1], true);
                            if line.count >= 5 && line.space_count == 0 {
                                return Some(p);
                            }
                        }
                    }
                }
            }
        }
        None
    }

    pub fn eval_all(&mut self, player: u8) -> u32 {
        let mut score = 0;
        for i in 0..self.height {
            for j in 0..self.width {
                if self.get(i as i32, j as i32) == Some(player) {
                    score += self.eval_pos(player, i, j);
                }
            }
        }
        score
    }

    fn connect_direction(
        &self,
        player: u8,
        row: usize,
        col: usize,
        dx: i32,
        dy: i32,
        consecutive: bool,
    ) -> Line {
        assert!(self.get(row as i32, col as i32) == Some(player));
        let mut cur_x = row as i32;
        let mut cur_y = col as i32;
        let mut flag = 1;
        let mut lefted_space = 0;
        let mut space_allow = if consecutive { 0 } else { 1 };
        let mut len = 1;
        let mut open_count = 2;
        let mut space_count = 0;
        let mut reversed = false;
        loop {
            loop {
                cur_x += dx * flag;
                cur_y += dy * flag;
                let cell = self.get(cur_x, cur_y);
                if cell == Some(0) {
                    if space_allow > 0 && self.get(cur_x + dx, cur_y + dy) == Some(player) {
                        space_allow -= 1;
                        space_count += 1;
                        continue;
                    } else {
                        let mut x = cur_x;
                        let mut y = cur_y;
                        while lefted_space <= 4 && self.get(x, y) == Some(0) {
                            lefted_space += 1;
                            x += dx * flag;
                            y += dy * flag;
                        }
                        break;
                    }
                } else if cell == Some(player) {
                    len += 1;
                } else {
                    assert!(cell.is_none() || cell != Some(player));
                    open_count -= 1;
                    break;
                }
            }
            if reversed {
                break;
            }
            reversed = true;
            flag = -1;
            cur_x = row as i32;
            cur_y = col as i32;
        }
        if len + space_count + lefted_space < 5 {
            open_count = 0;
        }
        if len >= 5 {
            if space_count == 0 {
                len = 5;
                open_count = 2;
            } else {
                len = 4;
                open_count = 1;
            }
        }

        Line::new(len, space_count, open_count)
    }

    fn connect_all_directions(&self, player: u8, row: usize, col: usize) -> Vec<Line> {
        let dirs = cfg::DIRS;
        let revs = cfg::REV_DIRS;
        let mut res = vec![];
        for i in 0..dirs.len() {
            let d = &dirs[i];
            let r = &revs[i];
            let l0 = self.connect_direction(player, row, col, d[0], d[1], true);
            // If we allow on space in row, different direction will generate different result,
            // One possible way is try to get the better one
            let l1 = self.connect_direction(player, row, col, d[0], d[1], false);
            let l2 = self.connect_direction(player, row, col, r[0], r[1], false);

            let l = if l1.cmp(&l2) == Ordering::Greater {
                l1
            } else {
                l2
            };
            let line = if l.cmp(&l0) == Ordering::Greater {
                l
            } else {
                l0
            };
            res.push(line);
        }
        res
    }

    pub fn eval_pos(&mut self, player: u8, row: usize, col: usize) -> u32 {
        let mut score = 0;
        let mut must_blocked = 0;
        let mut two_count = 0;
        let lines = self.connect_all_directions(player, row, col);
        assert!(lines.len() <= 4);
        for i in 0..lines.len() {
            let line = &lines[i];
            if line.is_winner_step() {
                return 100000;
            }
            if line.is_non_refutable() {
                return 5000;
            }
            if line.must_be_blocked() {
                must_blocked += 1;
            }
            if line.count >= 3 && line.open_count >= 2 {
                two_count += 1;
            }
            score += line.score();
        }
        if must_blocked >= 1 {
            score += must_blocked * 1000;
        }
        if two_count >= 2 {
            score += two_count * 100;
        }
        return score;
    }

    pub fn get(&self, row: i32, col: i32) -> Option<u8> {
        if !self.valid_pos(row, col) {
            None
        } else {
            Some(self.cells[row as usize][col as usize])
        }
    }

    fn get_prev(&self, row: i32, col: i32, dir: usize) -> Option<u8> {
        let revs = cfg::REV_DIRS;
        let p_row = row as i32 + revs[dir][0];
        let p_col = col as i32 + revs[dir][1];
        self.get(p_row, p_col)
    }

    fn valid_pos(&self, row: i32, col: i32) -> bool {
        row >= 0 && row < self.height as i32 && col >= 0 && col < self.width as i32
    }

    pub fn place(&mut self, row: usize, col: usize, player: u8) {
        self.cells[row][col] = player;
        if player != 0 {
            self.at_x = row as i32;
            self.at_y = col as i32;
        } else {
            self.at_x = -1;
            self.at_y = -1;
        }
    }

    pub fn is_remote_cell(&self, row: usize, col: usize) -> bool {
        let dirs = cfg::ALL_DIRS;
        for d in 1..3 {
            for k in 0..8 {
                let p = self.get(row as i32 + dirs[k][0] * d, col as i32 + dirs[k][1] * d);
                if !p.is_none() && p != Some(0) {
                    return false;
                }
            }
        }
        true
    }

    pub fn empty_cells_count(&self) -> u32 {
        let mut count = 0;
        for i in 0..self.height {
            for j in 0..self.width {
                if self.cells[i][j] == 0 {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn to_string(&self) -> String {
        let mut res = "".to_string();
        for i in 0..self.height {
            for j in 0..self.width {
                res += format!("{}", self.cells[i][j]).as_str()
            }
        }
        res
    }

    pub fn gen_ordered_moves_all(&mut self, player: u8) -> Vec<Move> {
        let mut moves = vec![];
        let mut row_min = self.height;
        let mut row_max = 0;
        let mut col_min = self.width;
        let mut col_max = 0;

        for i in 0..self.height {
            for j in 0..self.width {
                let n = self.get(i as i32, j as i32);
                if n.is_some() && n != Some(0) {
                    row_min = min(row_min, i);
                    row_max = max(row_max, i);
                    col_min = min(col_min, j);
                    col_max = max(col_max, j);
                }
            }
        }

        //let mut blocks = vec![];
        let mut max_score = 0;
        let mut win_step = -1;
        let mut lose_step = -1;
        let mut max_oppo = 0;
        for i in max(row_min as i32 - 1, 0) as usize..min(self.height, row_max + 2) {
            for j in max(col_min as i32 - 1, 0) as usize..min(self.width, col_max + 2) {
                if self.get(i as i32, j as i32) != Some(0) || self.is_remote_cell(i, j) {
                    continue;
                }
                self.place(i, j, player);
                let mut score = self.eval_pos(player, i, j) as i32;
                self.place(i, j, cfg::opponent(player));
                let oppo_score = self.eval_pos(cfg::opponent(player), i, j) as i32;
                if score >= 100000 {
                    win_step = moves.len() as i32;
                }
                if oppo_score >= 100000 {
                    lose_step = moves.len() as i32;
                }
                max_score = std::cmp::max(max_score, score);
                max_oppo = std::cmp::max(max_oppo, oppo_score);
                if oppo_score >= 5000 && score <= 2000 {
                    score = oppo_score;
                }
                self.place(i, j, 0);
                moves.push(Move::new(i, j, score as i32, oppo_score as i32));
            }
        }
        if win_step != -1 {
            return vec![moves[win_step as usize]];
        }
        if lose_step != -1 {
            return vec![moves[lose_step as usize]];
        }
        moves.sort_by(|a, b| {
            if a.score != b.score {
                b.score.cmp(&a.score)
            } else {
                b.original_score.cmp(&a.original_score)
            }
        });
        moves
    }

    pub fn gen_ordered_moves(&mut self, player: u8) -> Vec<Move> {
        let mut moves = vec![];
        let mut row_min = self.height;
        let mut row_max = 0;
        let mut col_min = self.width;
        let mut col_max = 0;

        for i in 0..self.height {
            for j in 0..self.width {
                let n = self.get(i as i32, j as i32);
                if n.is_some() && n != Some(0) {
                    row_min = min(row_min, i);
                    row_max = max(row_max, i);
                    col_min = min(col_min, j);
                    col_max = max(col_max, j);
                }
            }
        }

        for i in max(row_min as i32 - 1, 0) as usize..min(self.height, row_max + 2) {
            for j in max(col_min as i32 - 1, 0) as usize..min(self.width, col_max + 2) {
                if self.get(i as i32, j as i32) != Some(0) || self.is_remote_cell(i, j) {
                    continue;
                }
                self.place(i, j, player);
                let score = self.eval_pos(player, i, j);
                self.place(i, j, 0);
                moves.push(Move::new(i, j, score as i32, score as i32));
            }
        }
        moves.sort_by(|a, b| b.score.cmp(&a.score));
        moves
    }

    pub fn print(&self) {
        use yansi::Paint;

        for i in 0..self.height {
            let mut res = "".to_string();
            for j in 0..self.width {
                let last_placed = i == self.at_x as usize && j == self.at_y as usize;
                let cell = match self.cells[i][j] {
                    1 => Paint::green(if last_placed { " O" } else { " o" }),
                    2 => Paint::red(if last_placed { " X" } else { " +" }),
                    _ => Paint::white(" ."),
                };
                res += format!("{}", cell).as_str();
            }
            println!("{}", res.as_str());
        }
    }

    pub fn print_debug(&self, moves: &Vec<Move>, score: &Vec<Vec<usize>>, best: &Move) {
        use yansi::Paint;

        for i in 0..self.height {
            let mut res = "".to_string();
            for j in 0..self.width {
                let mut found = moves.len();
                for k in 0..moves.len() {
                    if i == moves[k].x && j == moves[k].y {
                        found = k;
                        break;
                    }
                }
                if found != moves.len() {
                    let w = score[found][0];
                    let l = score[found][1];
                    //let r = format!("{:.1}", w as f32 * 100.0 / (w + l) as f32);
                    let r = format!("{}/{}", w, l);
                    if i == best.x && j == best.y {
                        res += format!("{: ^6}", Paint::yellow(r)).as_str();
                    } else {
                        res += format!("{: ^6}", Paint::blue(r)).as_str();
                    }
                    continue;
                } else {
                    let last_placed = i == self.at_x as usize && j == self.at_y as usize;
                    let cell = match self.cells[i][j] {
                        1 => Paint::green(if last_placed { " O" } else { " o" }),
                        2 => Paint::red(if last_placed { " X" } else { " +" }),
                        _ => Paint::white(" ."),
                    };
                    res += format!("{: ^6}", cell).as_str();
                }
            }
            println!("{}", res.as_str());
        }
    }

    pub fn new_default() -> Board {
        let mut res = String::from("");
        for _ in 0..(15 * 15) {
            res = res + "0";
        }
        Board::from(res)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Move {
    pub x: usize,
    pub y: usize,
    pub score: i32,
    pub original_score: i32,
}

impl Move {
    pub fn new(x: usize, y: usize, score: i32, original_score: i32) -> Self {
        Self {
            x: x,
            y: y,
            score: score,
            original_score: original_score,
        }
    }

    pub fn is_threaten(&self) -> bool {
        self.score >= 1000
    }

    pub fn is_dead_move(&self) -> bool {
        self.score >= 100000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_create() {
        let board = Board::new(String::from("121211"), 1, 6);
        assert_eq!(board.width, 1);
        assert_eq!(board.height, 6);
        assert_eq!(board.to_string(), "121211");
    }

    #[test]
    fn test_board_copy() {
        let board = Board::new(String::from("121211"), 1, 6);
        let mut copy = board.clone();
        copy.place(0, 0, 0);
        assert_eq!(board.get(0, 0), Some(1));
        assert_eq!(copy.get(0, 0), Some(0));
    }

    #[test]
    fn test_line() {
        let mut line = Line::new(5, 0, 0);
        assert_eq!(line.is_non_refutable(), false);

        line = Line::new(5, 0, 1);
        assert_eq!(line.is_non_refutable(), false);

        line = Line::new(5, 1, 0);
        assert_eq!(line.is_non_refutable(), false);
    }

    #[test]
    fn test_line_compare() {
        let line1 = Line::new(3, 0, 1);
        let line2 = Line::new(4, 0, 1);
        assert_eq!(line1.cmp(&line2), Ordering::Less);

        let line3 = Line::new(3, 1, 2);
        assert_eq!(line1.cmp(&line3), Ordering::Less);

        let line4 = Line::new(3, 0, 1);
        assert_eq!(line1.cmp(&line4), Ordering::Equal);

        let line5 = Line::new(5, 0, 0);
        assert_eq!(line1.cmp(&line5), Ordering::Less);
    }

    #[test]
    fn test_board_from_string() {
        let board = Board::from(String::from("1212112121121211212112121"));
        assert_eq!(board.width, 5);
        assert_eq!(board.height, 5);
    }

    #[test]
    #[should_panic(expected = "Invalid input string size")]
    fn test_board_from_string_panic() {
        let _ = Board::from(String::from("12123"));
    }

    #[test]
    #[should_panic(expected = "Invalid board size with 2*2 <> 2")]
    fn test_board_size_validation() {
        Board::new(String::from("12"), 2, 2);
    }

    #[test]
    fn test_board_elements() {
        let board = Board::new(String::from("000112000112"), 6, 2);
        assert_eq!(board.cells[0][0], 0);
        assert_eq!(board.cells[0][1], 0);
        assert_eq!(board.cells[1][4], 1);
        assert_eq!(board.cells[1][5], 2);
    }

    #[test]
    fn test_board_check_winner() {
        let mut board = Board::new(String::from("1111100000"), 5, 2);
        assert_eq!(board.any_winner(), Some(1));

        board = Board::new(String::from("1111000000"), 5, 2);
        assert_eq!(board.any_winner(), None);

        board = Board::new(String::from("1111022222"), 5, 2);
        assert_eq!(board.any_winner(), Some(2));

        board = Board::new(String::from("111101111011110"), 5, 3);
        assert_eq!(board.any_winner(), None);

        board = Board::new(String::from("1111011110111101111010000"), 5, 5);
        assert_eq!(board.any_winner(), Some(1));

        board = Board::new(String::from("1111011110111100111010000"), 5, 5);
        assert_eq!(board.any_winner(), None);

        board = Board::new(String::from("10000 01000 00100 00010 00001"), 5, 5);
        assert_eq!(board.any_winner(), Some(1));

        board = Board::new(String::from("10000 01000 00000 00010 00001"), 5, 5);
        assert_eq!(board.any_winner(), None);

        board = Board::new(String::from("22220 10000 01000 00100 00010 00001"), 5, 6);
        assert_eq!(board.any_winner(), Some(1));

        board = Board::new(String::from("22220 00001 00010 00100 01000 10000"), 5, 6);
        assert_eq!(board.any_winner(), Some(1));

        board = Board::new(String::from("22222 00001 00010 00100 01000 10000"), 5, 6);
        assert_eq!(board.any_winner(), Some(2));

        board = Board::new(
            String::from(
                "
        10000
        10001
        01021
        00001
        00001
        00001",
            ),
            5,
            6,
        );

        let line = board.connect_direction(1, 1, 4, 1, 0, true);
        assert_eq!(line.count, 5);
        assert_eq!(board.any_winner(), Some(1));
    }

    #[test]
    fn test_board_score() {
        let mut board = Board::new(String::from("1111020000"), 5, 2);
        assert_eq!(board.eval_pos(1, 0, 0), 1050);
        assert_eq!(board.eval_pos(2, 1, 0), 0);

        board = Board::new(String::from("1111111111"), 5, 2);
        assert_eq!(board.eval_all(2), 0);
        assert_eq!(board.eval_all(1), 1000000);

        board = Board::new(String::from("10000 01000 00100"), 5, 3);
        assert_eq!(board.eval_all(1), 0);

        board = Board::new(
            String::from(
                "
        10000
        01000
        00100
        00000",
            ),
            5,
            4,
        );
        assert_eq!(board.eval_all(1), 0);

        board = Board::new(
            String::from(
                "
        10000
        01000
        00100
        00000
        00000",
            ),
            5,
            5,
        );
        assert_eq!(board.eval_all(1), 90);

        board = Board::new(
            String::from(
                "
        10000
        01100
        00100
        00000",
            ),
            5,
            4,
        );
        assert_eq!(board.eval_all(1), 50);

        board = Board::new(
            String::from(
                "
            000000
            011100
            011100
            011100
            000000
            000000",
            ),
            6,
            6,
        );
        assert_eq!(board.eval_all(1), 27750);

        board = Board::new(
            String::from(
                "
        101100
        000000",
            ),
            6,
            2,
        );
        assert_eq!(board.eval_all(1), 75);

        board = Board::new(
            String::from(
                "
        1011100
        0000000",
            ),
            7,
            2,
        );
        assert_eq!(board.eval_all(1), 4180);

        board = Board::new(
            String::from(
                "
        0000000
        0001000
        0001000
        0022000
        0000000
        ",
            ),
            7,
            5,
        );
        assert_eq!(board.eval_all(1), 0);

        board = Board::new(
            String::from(
                "
        0000000
        0001000
        0001000
        0111000
        0000000
        ",
            ),
            7,
            5,
        );
        assert_eq!(board.eval_pos(1, 3, 3), 2300);

        board = Board::new(
            String::from(
                "
        0000000
        0001000
        0001000
        0020000
        0000000
        ",
            ),
            7,
            5,
        );
        assert_eq!(board.eval_all(1), 50);

        board = Board::new(
            String::from(
                "
        0000000
        0001000
        0001000
        0020000
        0000000
        0000000
        ",
            ),
            7,
            6,
        );
        assert_eq!(board.eval_all(1), 50);

        board = Board::new(
            String::from(
                "
        0000000
        0000000
        0001000
        0001000
        0001200
        0002000
        0000000
        ",
            ),
            7,
            7,
        );
        assert_eq!(board.eval_pos(1, 2, 3), 30);

        board = Board::new(
            String::from(
                "
        0000000
        0000000
        0000000
        0001100
        0001200
        0002000
        0000000
        ",
            ),
            7,
            7,
        );
        assert_eq!(board.eval_pos(1, 3, 4), 50);

        board = Board::new(
            String::from(
                "
        0000000
        0001000
        0001000
        0001000
        0001000
        0002000
        0002000
        ",
            ),
            7,
            7,
        );
        assert_eq!(board.eval_pos(1, 1, 3), 1050);
    }

    #[test]
    fn test_one_direction_corner_case() {
        let mut board = Board::new(
            String::from(
                "
        0000000
        0000100
        0000000
        0000000
        0000000
        0000000
        ",
            ),
            7,
            6,
        );

        let line = board.connect_direction(1, 1, 4, 1, 1, true);
        assert_eq!(line, Line::new(1, 0, 0));
        let line = board.connect_direction(1, 1, 4, 1, 1, false);
        assert_eq!(line, Line::new(1, 1, 0));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00000100
        00000000
        00000000
        00000000
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 5, 1, 1, true);
        assert_eq!(line, Line::new(1, 0, 2));
        let line = board.connect_direction(1, 2, 5, 1, 1, false);
        assert_eq!(line, Line::new(1, 1, 2));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00000100
        00000010
        00000000
        00000000
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 5, 1, 1, true);
        assert_eq!(line, Line::new(2, 0, 2));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00000100
        00000010
        00000000
        00000000
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 5, 1, 1, true);
        assert_eq!(line, Line::new(2, 0, 2));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00000100
        00000010
        00000002
        00000000
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 5, 1, 1, true);
        assert_eq!(line, Line::new(2, 0, 0));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00001000
        00000100
        00000010
        00000002
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 4, 1, 1, true);
        assert_eq!(line, Line::new(3, 0, 1));

        let line = board.connect_direction(1, 3, 5, 1, 1, true);
        assert_eq!(line, Line::new(3, 0, 1));

        board = Board::new(
            String::from(
                "
        00000000
        00000000
        00201020
        00000000
        00000000
        00000000
        ",
            ),
            8,
            6,
        );

        let line = board.connect_direction(1, 2, 4, 0, 1, true);
        assert_eq!(line, Line::new(1, 0, 0));
    }
}
