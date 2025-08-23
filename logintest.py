import os
from dotenv import load_dotenv

from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()  

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")  # get from env


def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
    conn.commit()
    conn.close()


@app.route("/")
def home():
    if "username" in session:
        return f"Welcome, {session['username']}! <br><a href='/logout'>Logout</a>"
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["username"] = username
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password!", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
