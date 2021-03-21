#![allow(dead_code)]
use super::algo::*;
use super::board::*;
use super::utils::*;

#[cfg(feature = "random")]
use rand::Rng;
use std::env;

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
        let next_player = cfg::opponent(player);
        self.new_node(index, board.clone(), next_player, Some(mv))
    }

    fn backpropagete(&mut self, index: Id, winner: Option<u8>) {
        self.nodes[index].visited_count += 1;
        let player = self.nodes[index].player;
        //println!("backpropgate: {} {:?}", index, winner);
        if winner == Some(player) {
            self.nodes[index].loss_count += 1;
        } else if winner == Some(cfg::opponent(player)) {
            self.nodes[index].win_count += 1;
        }
        if !self.nodes[index].is_root() {
            self.backpropagete(self.nodes[index].parent, winner);
        }
    }

    fn rollout_policy(&self, moves: &Vec<Move>) -> Option<Move> {
        if moves.len() == 0 {
            return None;
        }
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
                Some(moves[idx])
            } else {
                Some(moves[0])
            }
        }
    }

    pub fn rollout(&mut self, index: Id) -> Option<u8> {
        let mut current_state = self.nodes[index].state.clone();
        let mut player = self.nodes[index].player;
        loop {
            let moves = current_state.gen_ordered_moves_all(player);
            let rollout_move = self.rollout_policy(&moves);
            if rollout_move.is_none() {
                return None;
            }
            let mv = rollout_move.unwrap();
            current_state.place(mv.x, mv.y, player);
            player = cfg::opponent(player);
            if mv.is_dead_move() {
                let winner = current_state.any_winner();
                if winner.is_some() {
                    return winner;
                }
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
    debug: bool,
}

impl MonteCarlo {
    pub fn new(state: Board, player: u8, simulate_count: u32) -> Self {
        let mut s = Self {
            tree: Tree::new(),
            simulate_count: simulate_count,
            debug: match env::var("GOMOKU_DEBUG") {
                Ok(_) => true,
                _ => false,
            },
        };
        s.tree.new_node(0, state, player, None);
        s
    }

    fn get(&self, index: usize) -> &Node {
        self.tree.get_node(index).unwrap()
    }

    fn tree_policy(&mut self) -> usize {
        let mut cur = 0;
        while !self.get(cur).is_terminal_node() {
            if !self.get(cur).is_fully_expanded() {
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
            if self.debug {
                if i % 100 == 0 {
                    println!(
                        "%{:.2} {} : rollout {}",
                        (i as f32 * 100.0) / self.simulate_count as f32,
                        v,
                        self.tree.nodes.len()
                    );
                }
            }
            let r = self.tree.rollout(v);
            self.tree.backpropagete(v, r);
        }
        let best = self.tree.best_child(0);
        let res = self.get(best).action.unwrap();
        self.print_debug(best, &res);
        res
    }

    fn print_debug(&self, best: usize, mv: &Move) {
        if self.debug {
            println!("len: {} best: {}", self.get(0).children.len(), best);
            let mut score = vec![];
            let mut moves = vec![];
            for x in self.get(0).children.iter() {
                println!("candidte move: {:?}", self.get(*x).action);
                moves.push(self.tree.nodes[*x].action.unwrap());
                score.push(vec![
                    self.get(*x).win_count as usize,
                    self.get(*x).loss_count as usize,
                ]);
            }
            println!("best move: {:?}", mv);
            self.tree.nodes[best].state.print_debug(&moves, &score, &mv);
            println!("win: {}", self.tree.nodes[best].win_count);
            println!("loss: {}", self.tree.nodes[best].loss_count);
            println!("visited: {}", self.tree.nodes[best].visited_count);
        }
    }
}

impl GomokuSolver for MonteCarlo {
    fn best_move(input: &str) -> Move {
        let board = Board::new(input.to_string(), 15, 15);
        let player = board.next_player();
        let mut monte = MonteCarlo::new(board, player, 2000);
        monte.search_move()
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

        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 8);
        root.expand(1);
        assert_eq!(root.get_node(1).unwrap().untried_moves.len(), 7);
    }

    #[test]
    fn test_tree() {
        let mut root = Tree::new();
        assert_eq!(root.nodes.len(), 0);
        let node = root.new_node(0, Board::new_default(), 1, None).unwrap();
        assert_eq!(node.parent, 0);
        assert_eq!(node.index, 0);
        assert_eq!(node.is_root(), true);

        root.nodes[0].untried_moves.push(Move::new(1, 2, 0, 0));
        let node_2 = root.expand(0).unwrap();
        assert_eq!(node_2.parent, 0);
        assert_eq!(root.get_node(0).unwrap().children.len(), 1);
        assert_eq!(root.get_node(1).unwrap().children.len(), 0);
        assert_eq!(root.get_node(1).unwrap().player, 2);
        let mv = root.get_node(1).unwrap().action.unwrap();
        assert_eq!(mv.x, 1);
        assert_eq!(mv.y, 2);
        root.backpropagete(1, Some(1));
        assert_eq!(root.get_node(1).unwrap().loss_count, 0);
        assert_eq!(root.get_node(1).unwrap().win_count, 1);

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
}
