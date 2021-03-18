#![allow(dead_code)]
use super::board::*;
use super::utils::*;
#[cfg(feature = "random")]
use rand::Rng;

type Id = usize;

pub struct Tree {
    nodes: Vec<Node>,
}

#[derive(Debug)]
pub struct Node {
    parent: Id,
    index: Id,
    children: Vec<Id>,
    visited_count: u32,
    win_count: u32,
    loss_count: u32,
    state: Board,
    untried_moves: Vec<Move>,
    player: u8,
    action: Option<Move>,
}

impl Tree {
    pub fn new() -> Self {
        Tree { nodes: vec![] }
    }

    pub fn new_node(
        &mut self,
        parent: Id,
        state: Board,
        player: u8,
        mv: Option<Move>,
    ) -> Option<&Node> {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, parent, state, player, mv));
        if parent != id {
            self.nodes[parent].children.push(id);
        }
        self.nodes.get(id)
    }

    fn get_node(&self, id: Id) -> Option<&Node> {
        self.nodes.get(id)
    }

    fn expand(&mut self, index: Id) -> Option<&Node> {
        let parent = self.nodes.get(index).unwrap();
        let player = parent.player;
        let mv = parent.untried_moves.first().unwrap().clone();
        self.nodes[index].untried_moves.remove(0);
        let mut board = self.nodes[index].state.clone();
        board.place(mv.x, mv.y, player);
        self.new_node(
            index,
            board,
            cfg::opponent(player),
            if index == 0 { Some(mv) } else { None },
        )
    }

    fn backpropagete(&mut self, index: Id, winner: Option<u8>) {
        self.nodes[index].visited_count += 1;
        let player = self.nodes[index].player;
        if winner == Some(player) {
            self.nodes[index].loss_count += 1;
        } else if winner == Some(cfg::opponent(player)) {
            self.nodes[index].win_count += 1;
        }
        if !self.nodes[index].is_root() {
            self.backpropagete(self.nodes[index].parent, winner);
        }
    }

    fn rollout_policy(&self, moves: &Vec<Move>) -> Move {
        cfg_if::cfg_if! {
            if #[cfg(feature = "random")] {
                let mut rng = rand::thread_rng();
                let mut idx = 0;
                loop {
                    if idx + 1 < moves.len()
                    && moves[idx].score == moves[idx + 1].score
                    && moves[idx].original_score == moves[idx + 1].original_score
                    {
                        idx += 1;
                    } else {
                        break;
                    }
                }
                let idx = rng.gen_range(0..idx + 1);
                moves[idx].clone()
            } else {
                moves[0].clone()
            }
        }
    }

    pub fn rollout(&mut self, index: Id) -> Option<u8> {
        let mut current_state = self.nodes[index].state.clone();
        let mut player = self.nodes[index].player;
        loop {
            let w = current_state.any_winner();
            if w.is_some() {
                return w;
            } else {
                let moves = current_state.gen_ordered_moves_all(player);
                if moves.len() == 0 {
                    return None;
                }
                let mv = self.rollout_policy(&moves);
                current_state.place(mv.x, mv.y, player);
                player = cfg::opponent(player);
            }
        }
    }

    pub fn best_child(&self, index: usize) -> usize {
        //let c_param = 0.5;
        let c_param = 0.7;
        let mut res = 0;
        let mut cur_max = f64::MIN;
        let node = &self.nodes[index];
        for i in 0..node.children.len() {
            let c = &self.nodes[node.children[i]];
            let r = (2.0 * node.n().ln() / c.n()).sqrt();
            let v = c.q() / c.n() + c_param * r;
            if v > cur_max {
                cur_max = v;
                res = c.index;
            }
        }
        return res;
    }

    pub fn best_move(&self, index: usize) -> usize {
        let mut res = -1;
        let mut cur_max = f64::MIN;
        let node = &self.nodes[index];
        for i in 0..node.children.len() {
            let c = &self.nodes[node.children[i]];
            let v = c.win_count as f64 / c.visited_count as f64;
            if v > cur_max && c.visited_count >= 8 {
                cur_max = v;
                res = c.index as i32;
            }
        }
        if res == -1 {
            self.nodes[index].children[0]
        } else {
            res as usize
        }
    }
}

impl Node {
    pub fn new(index: Id, parent: Id, state: Board, player: u8, mv: Option<Move>) -> Self {
        let mut s = Node {
            visited_count: 0,
            win_count: 0,
            loss_count: 0,
            state: state.clone(),
            untried_moves: vec![],
            children: vec![],
            player: player,
            parent: parent,
            index: index,
            action: mv,
        };
        s.untried_moves = s.state.gen_ordered_moves_all(player);
        s
    }

    pub fn is_root(&self) -> bool {
        self.parent == self.index
    }

    pub fn n(&self) -> f64 {
        self.visited_count as f64
    }

    pub fn q(&self) -> f64 {
        (self.win_count as i32 - self.loss_count as i32) as f64
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.untried_moves.len() == 0
    }

    pub fn is_terminal_node(&self) -> bool {
        self.state.any_winner() != None
    }
}

pub struct MonteCarlo {
    tree: Tree,
    simulate_count: u32,
}

impl MonteCarlo {
    pub fn new(state: Board, player: u8, simulate_count: u32) -> Self {
        let mut s = Self {
            tree: Tree::new(),
            simulate_count: simulate_count,
        };
        s.tree.new_node(0, state, player, None);
        s
    }

    fn tree_policy(&mut self) -> usize {
        let mut cur = 0;
        while !self.tree.nodes[cur].is_terminal_node() {
            if !self.tree.nodes[cur].is_fully_expanded() {
                let n = self.tree.expand(cur);
                return n.unwrap().index;
            } else {
                cur = self.tree.best_child(cur);
            }
        }
        return cur;
    }

    pub fn search_move(&mut self) -> Move {
        for i in 0..self.simulate_count {
            let v = self.tree_policy();
            if i % 100 == 0 {
                println!(
                    "%{:.2} {} : rollout {}",
                    (i as f32 * 100.0) / self.simulate_count as f32,
                    v,
                    self.tree.nodes.len()
                );
            }
            let r = self.tree.rollout(v);
            self.tree.backpropagete(v, r);
        }
        let best = self.tree.best_move(0);
        println!("len: {} best: {}", self.tree.nodes[0].children.len(), best);
        let mut score = vec![];
        let mut moves = vec![];
        for x in self.tree.nodes[0].children.iter() {
            println!("candidte move: {:?}", self.tree.nodes[*x].action);
            moves.push(self.tree.nodes[*x].action.unwrap());
            score.push(vec![
                self.tree.nodes[*x].win_count as usize,
                self.tree.nodes[*x].loss_count as usize,
            ]);
        }
        let res = self.tree.nodes[best].action.unwrap();
        println!("best move: {:?}", res);
        self.tree.nodes[best]
            .state
            .print_debug(&moves, &score, &res);
        println!("win: {}", self.tree.nodes[best].win_count);
        println!("loss: {}", self.tree.nodes[best].loss_count);
        println!("visited: {}", self.tree.nodes[best].visited_count);
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_create() {
        let mut board = Board::new_default();
        board.place(5, 5, 1);
        let mut root = Tree::new();
        root.new_node(0, board, 1, None).unwrap();
        assert_eq!(root.get_node(0).unwrap().parent, 0);

        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 8);

        root.expand(0);
        assert_eq!(root.get_node(0).unwrap().parent, 0);
        assert_eq!(root.get_node(0).unwrap().untried_moves.len(), 7);
        assert_eq!(root.get_node(1).unwrap().parent, 0);
        root.backpropagete(1, Some(2));

        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 12);
        root.expand(1);
        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 11);
    }

    #[test]
    fn test_tree() {
        let mut root = Tree::new();
        assert_eq!(root.nodes.len(), 0);
        let node = root.new_node(0, Board::new_default(), 1, None).unwrap();
        assert_eq!(node.parent, 0);
        assert_eq!(node.index, 0);
        assert_eq!(node.is_root(), true);
        let node_2 = root.new_node(0, Board::new_default(), 2, None).unwrap();
        assert_eq!(node_2.parent, 0);
        assert_eq!(root.get_node(0).unwrap().children.len(), 1);
        assert_eq!(root.get_node(1).unwrap().children.len(), 0);
        root.backpropagete(1, Some(1));
        assert_eq!(root.get_node(0).unwrap().loss_count, 1);
        assert_eq!(root.get_node(0).unwrap().win_count, 0);
    }

    #[test]
    fn test_rollout() {
        let mut board = Board::new_default();
        board.place(5, 5, 1);
        let mut root = Tree::new();
        root.new_node(0, board, 2, None).unwrap();
        assert_eq!(root.get_node(0).unwrap().parent, 0);
        assert_eq!(root.get_node(0).unwrap().player, 2);

        let res = root.rollout(0);
        assert_eq!(res.is_some(), true);
    }

    #[test]
    fn test_monte_carlo_basic() {
        let mut board = Board::new_default();
        board.place(7, 7, 1);
        let mut monte_carlo = MonteCarlo::new(board, 2, 20);
        assert_eq!(monte_carlo.tree.nodes[0].is_fully_expanded(), false);
        let mv = monte_carlo.search_move();
        println!("{:?}", mv);
        assert!((mv.x as i32 - 7 as i32).abs() <= 1 && (mv.y as i32 - 7 as i32).abs() <= 1);
    }

    #[test]
    fn test_monte_block_one() {
        let board = Board::new(
            String::from(
                "000000000000000
                 000000000000000
                 000000000000000
                 000000002212000
                 000000012210000
                 000000021112000
                 000000112200000
                 000000000100000
                 000000001000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000",
            ),
            15,
            15,
        );

        board.print();
        let mut monte = MonteCarlo::new(board.clone(), 2, 100);
        let mv = monte.search_move();
        let row = mv.x;
        let col = mv.y;
        assert!(col == 10 && (row == 2 || row == 7));
    }

    #[test]
    fn test_monte_block_three_full() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000001220000000
                 000000212000000
                 000002011000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000
                 000000000000000",
            ),
            15,
            15,
        );

        board.print();
        let moves = board.gen_ordered_moves_all(1);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board, 1, 2000);
        println!("children: {}", monte.tree.nodes[0].untried_moves.len());
        assert_eq!(monte.tree.nodes[0].untried_moves.len(), 20);
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), false);
        let mv = monte.search_move();
        let row = mv.x;
        let col = mv.y;
        assert!((col == 4 || col == 8) && (row == 5 || row == 9));
    }

    #[test]
    fn test_monte_block_four() {
        let board = Board::new(
            String::from(
                "000020200000000
                 000001100000000
                 000000100000000
                 000000110010000
                 000000021101000
                 000000012222000
                 000000212210000
                 000000012220000
                 000000012201000
                 000000021100000
                 000000200200000
                 000001000000000
                 000000000000000
                 000000000000000
                 000000000000000",
            ),
            15,
            15,
        );

        board.print();
        let mut monte = MonteCarlo::new(board.clone(), 1, 100);
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), false);
        assert_eq!(monte.tree.nodes[0].is_terminal_node(), false);
        let mv = monte.search_move();
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), true);
        for x in 0..monte.tree.nodes[0].children.len() {
            let c = monte.tree.nodes[0].children[x];
            println!("candidte state move: {:?}", monte.tree.nodes[c].action);
            monte.tree.nodes[c].state.print();
        }
        let row = mv.x;
        let col = mv.y;
        println!("move: {:?}", mv);
        assert!(row == 5 && col == 12);
    }

    #[test]
    fn test_monte_block_five() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000001000000000
                000000200000000
                000022120000000
                000000102000000
                000000211201000
                000000121010000
                000000010100000
                000000000020000
                000000020000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        let moves = board.gen_ordered_moves_all(2);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board.clone(), 2, 2000);
        let mv = monte.search_move();
        board.place(mv.x, mv.y, 2);
        board.print();
        let row = mv.x;
        let col = mv.y;
        println!("{:?}", mv);
        assert!(row == 5 && col == 5);
    }

    #[test]
    fn test_monte_block_five_orig() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000000000000000
                000000000000000
                000002020000000
                000000102000000
                000000211201000
                000000121010000
                000000010100000
                000000000020000
                000000020000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        let moves = board.gen_ordered_moves_all(2);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board.clone(), 2, 2000);
        let mv = monte.search_move();
        board.place(mv.x, mv.y, 2);
        board.print();
        let row = mv.x;
        let col = mv.y;
        println!("{:?}", mv);
        assert!(row == 5 && col == 12);
    }

    #[test]
    fn test_monte_block_three_two() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000200000
                000000211000000
                000000010000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        board.print();
        let moves = board.gen_ordered_moves_all(2);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board.clone(), 2, 2000);
        let mv = monte.search_move();
        println!("move: {:?}", mv);
        let row = mv.x;
        let _col = mv.y;
        assert!(row == 4 || row == 5 || row == 6 || row == 7);
        //assert!(row == 4 && col == 8);
    }

    #[test]
    fn test_monte_block_three_bug() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000000000000000
                000001000000000
                000000201000000
                000001220110000
                000022212200000
                000112012200000
                000012120010000
                000001201000000
                000021210000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        board.print();
        let moves = board.gen_ordered_moves_all(1);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board.clone(), 1, 2000);
        let mv = monte.search_move();
        println!("left: {}", monte.tree.nodes[1].untried_moves.len());
        println!("move: {:?}", mv);
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), true);
        let row = mv.x;
        let col = mv.y;
        assert!((row == 3 && col == 6) || (row == 2 && col == 6));
    }

    #[test]
    fn test_monte_block_three_trivial_bug() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000000000000000
                000000000000000
                000002000000000
                000000100000000
                000001012200000
                000022211000000
                000000112100000
                000011121020000
                000011020200000
                000220000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        board.print();
        let moves = board.gen_ordered_moves_all(2);
        for x in 0..moves.len() {
            println!("{:?}", moves[x]);
        }
        let mut monte = MonteCarlo::new(board.clone(), 2, 2000);
        let mv = monte.search_move();
        println!("left: {}", monte.tree.nodes[1].untried_moves.len());
        println!("move: {:?}", mv);
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), true);
        let row = mv.x;
        let col = mv.y;
        assert!(row == 11 && col == 5);
    }

    #[test]
    fn test_monte_block_win_step() {
        let mut board = Board::new(
            String::from(
                "000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000200000000
                000021110000000
                000002210000000
                000000120000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000
                000000000000000",
            ),
            15,
            15,
        );
        board.print();
        let mut monte = MonteCarlo::new(board.clone(), 1, 2000);
        let mv = monte.search_move();
        println!("left: {}", monte.tree.nodes[1].untried_moves.len());
        println!("move: {:?}", mv);
        assert_eq!(monte.tree.nodes[0].is_fully_expanded(), true);
        let row = mv.x;
        let col = mv.y;
        board.place(row, col, 1);
        board.print();
        println!("{:?}", mv);
        assert!(row == 7 && col == 8);
    }
}

/*
player 1
000000000000000
000000000000000
000000000000000
000000000000000
000000000000000
000000001100000
000000021200000
000000212100000
000000212100000
000000122000000
000000000000000
000000000000000
000000000000000
000000000000000
000000000000000 */

/*
player: 2
000000000000000
000000000000000
000000000000000
000000000000000
000002000000000
000000100000000
000001012200000
000022211000000
000000112100000
000011121020000
000011020200000
000220000000000
000000000000000
000000000000000
000000000000000
*/
