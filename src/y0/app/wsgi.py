# -*- coding: utf-8 -*-

"""A web application exposing y0 functionality."""

import os
from typing import Optional, Sequence

import flask
import flask_bootstrap
import pyparsing
from flask import flash, render_template
from flask_wtf import FlaskForm
from wtforms import RadioField, StringField, SubmitField

from y0.dsl import Expression, Variable, _upgrade_variables
from y0.mutate.canonicalize import canonicalize
from y0.parser import parse_causaleffect, parse_craig, parse_y0

app = flask.Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
app.config['SECRET_KEY'] = os.urandom(8)

flask_bootstrap.Bootstrap(app)

LANGUAGES = {
    'y0': parse_y0,
    'craig': parse_craig,
    'ce': parse_causaleffect,
}


class CanonicalizeForm(FlaskForm):
    """Form for parsing and canonicalizing a probability expression."""

    expression = StringField('Expression')
    ordering = StringField('Ordering')
    language = RadioField(
        'Language',
        default='y0',
        choices=[
            ('y0', 'y0 DSL'),
            ('craig', 'Craig-like'),
            ('ce', 'Causaleffect'),
        ],
    )
    submit = SubmitField('Parse')

    def parse_expression(self) -> Expression:
        """Parse the expression."""
        s = self.expression.data.strip().upper()
        return LANGUAGES[self.language.data](s)

    def parse_ordering(self) -> Optional[Sequence[Variable]]:
        """Parse the ordering."""
        if self.ordering.data:
            return _upgrade_variables(self.ordering.data.upper().split(','))


@app.route('/', methods=['GET', 'POST'])
def home():
    """Render the home page."""
    form = CanonicalizeForm()
    if form.is_submitted():
        try:
            expression = form.parse_expression()
        except pyparsing.ParseException:
            flash(f'Invalid input: {form.expression.data}')
        else:
            ordering = form.parse_ordering()
            if ordering is not None:
                try:
                    canonical_expression = canonicalize(expression, ordering)
                except KeyError as e:
                    flash(f'Missing variable in ordering: {e.args[0].name}')
                else:
                    flash(flask.Markup(
                        f'Input y0:     {expression.to_text()}<br />'
                        f'Canon y0: {canonical_expression.to_text()}',
                    ))
            else:
                flash(flask.Markup(
                    f'Input y0:     {expression.to_text()}<br />'
                ))

    return render_template('home.html', form=form)


if __name__ == '__main__':
    app.run()
